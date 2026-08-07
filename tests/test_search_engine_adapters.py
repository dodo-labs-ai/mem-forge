"""SearchEngine accepts adapters handles and routes channels through them."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memforge.config import DEFAULT_RANK_WINDOW_SIZE, DEFAULT_RRF_K, RetrievalConfig
from memforge.llm.structured import RerankResponse
from memforge.models import DocumentRecord, Memory, content_hash
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
from memforge.retrieval.search import SearchEngine, _quoted_identity_query
from memforge.storage.database import Database
from memforge.storage.adapters.protocols import EntityLinkCandidate, KeywordCandidate
from memforge.storage.adapters.sqlite import build_sqlite_adapters


class FakeCollection:
    def __init__(self, ids: list[str], distances: list[float] | None = None) -> None:
        self.ids = ids
        self.distances = distances or [0.01 for _ in ids]

    def query(self, **kwargs):
        return {"ids": [self.ids], "distances": [self.distances]}

    def upsert(self, **kwargs):
        pass

    def delete(self, **kwargs):
        pass

    def get(self, **kwargs):
        return {"ids": []}


class RecordingRerankClient:
    def __init__(self) -> None:
        self.prompt: str | None = None

    async def rerank_memories(self, prompt: str, **kwargs):
        self.prompt = prompt
        return RerankResponse(ranking=[0])


class QueryScoredKeyword:
    def __init__(self, *, full_score: float, quoted_score: float) -> None:
        self.full_score = full_score
        self.quoted_score = quoted_score

    async def search_metadata(self, query: str, *args, **kwargs):
        score = self.full_score if '"Find"' in query else self.quoted_score
        return [
            KeywordCandidate(
                memory_id="m-access-review",
                score=score,
                channel="bm25_metadata_tokens",
            )
        ]


def _memory(
    mem_id: str,
    content: str,
    status: str = "active",
    repo_identifier: str | None = None,
    memory_type: str = "fact",
) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type=memory_type,
        content=content,
        content_hash=content_hash(content),
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status=status,
        repo_identifier=repo_identifier,
    )


async def _document(
    db: Database,
    doc_id: str,
    source: str,
    *,
    client: str | None = None,
    title: str = "t",
) -> None:
    now = datetime.now(timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source=source,
            source_url=f"https://x/{doc_id}",
            title=title,
            space_or_project="PAY",
            author="a",
            last_modified=now,
            labels=[],
            version="1",
            content_hash=f"h-{doc_id}",
            token_count=1,
            raw_content_uri=None,
            raw_content_type="text/html",
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
            client=client,
        )
    )


async def _title_only_search_engine(
    db: Database,
) -> tuple[Memory, SearchEngine]:
    target = _memory("m-access-review", "Durable source-backed note with different wording")
    await db.insert_memory(target)
    await db.upsert_source(
        "src-teams",
        "teams",
        "PCC Agent Dev",
        "{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    await _document(
        db,
        "teams-access-review",
        "src-teams",
        title="Create Access Review in Quarterly Payroll",
    )
    await db.add_memory_source(
        target.id,
        "teams-access-review",
        "teams",
        None,
        source_updated_at=None,
    )
    adapters = build_sqlite_adapters(db, FakeCollection([]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]
    return target, engine


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "search-adapters.db"))
    await database.connect()
    for source_id in (
        "wiki",
        "jira",
        "other",
        "slack",
        "src-agent-codex",
        "src-jira",
        "src-mounttai",
        "src-other",
        "src-target",
        "src-wiki",
    ):
        await database.upsert_source(
            source_id,
            "test",
            source_id,
            "{}",
            "workspace",
            "dev",
            created_by_user_id="dev",
        )
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_search_routes_vector_and_bm25_through_the_adapters(db, monkeypatch):
    active = _memory("m-active", "PostgreSQL pooling memory")
    await db.insert_memory(active)

    adapters = build_sqlite_adapters(db, FakeCollection([active.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("PostgreSQL", top_k=10)
    assert [r.memory_id for r in result["results"]] == [active.id]


@pytest.mark.asyncio
async def test_search_exposes_rank_fusion_contributions_for_each_result(db):
    content_target = _memory("m-content-target", "PostgreSQL connection pooling guidance")
    vector_only = _memory("m-vector-only", "Unrelated operational note")
    await db.insert_memory(content_target)
    await db.insert_memory(vector_only)

    adapters = build_sqlite_adapters(
        db,
        FakeCollection(
            [vector_only.id, content_target.id],
            distances=[0.01, 0.02],
        ),
    )
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("PostgreSQL", top_k=10)

    evidence_by_id = {
        item.memory_id: item.retrieval_evidence or {}
        for item in result["results"]
    }
    assert evidence_by_id[content_target.id]["rank_fusion"] == {
        "intent": "known_item",
        "rrf_score": pytest.approx(
            0.15 / (DEFAULT_RRF_K + 2) + 0.25 / (DEFAULT_RRF_K + 1)
        ),
        "contributions": [
            {
                "channel": "vector",
                "rank": 2,
                "channel_weight": 0.15,
                "multiplier": 1.0,
                "weighted_score": pytest.approx(0.15 / (DEFAULT_RRF_K + 2)),
            },
            {
                "channel": "bm25_content",
                "rank": 1,
                "channel_weight": 0.25,
                "multiplier": 1.0,
                "weighted_score": pytest.approx(0.25 / (DEFAULT_RRF_K + 1)),
            },
        ],
    }
    assert evidence_by_id[vector_only.id]["rank_fusion"] == {
        "intent": "known_item",
        "rrf_score": pytest.approx(0.15 / (DEFAULT_RRF_K + 1)),
        "contributions": [
            {
                "channel": "vector",
                "rank": 1,
                "channel_weight": 0.15,
                "multiplier": 1.0,
                "weighted_score": pytest.approx(0.15 / (DEFAULT_RRF_K + 1)),
            },
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "requested_intent", "resolved_intent", "intent_source"),
    [
        ("edit a locked rule", None, "general_hybrid", "default"),
        ("编辑一个被别人锁住的规则", None, "general_hybrid", "default"),
        ("persist_cutoff_blocker_hints", None, "known_item", "deterministic"),
        ("SFPAY-179397", "general_hybrid", "general_hybrid", "requested"),
        ("how does this relate", "known_item", "known_item", "requested"),
    ],
)
async def test_search_resolves_ranked_intent_without_language_shape_routing(
    db,
    query,
    requested_intent,
    resolved_intent,
    intent_source,
):
    memory = _memory("m-intent", "Rule locking behavior")
    await db.insert_memory(memory)
    adapters = build_sqlite_adapters(db, FakeCollection([memory.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda value: [0.1]

    result = await engine.search(query, intent=requested_intent, top_k=1)

    assert result["retrieval_intent"] == {
        "requested_intent": requested_intent,
        "resolved_intent": resolved_intent,
        "intent_source": intent_source,
        "fallback_reason": None,
    }
    assert result["query_analysis"]["strategies_used"] == [
        "vector",
        "bm25_content",
        "bm25_metadata_tokens",
    ]
    evidence = result["results"][0].retrieval_evidence or {}
    assert evidence["rank_fusion"]["intent"] == resolved_intent
    assert "profile" not in evidence["rank_fusion"]


@pytest.mark.asyncio
async def test_requested_relationship_falls_back_without_traversable_entity(db):
    memory = _memory("m-fallback", "General relationship note")
    await db.insert_memory(memory)
    adapters = build_sqlite_adapters(db, FakeCollection([memory.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda value: [0.1]

    result = await engine.search("how is this related", intent="relationship", top_k=1)

    assert result["retrieval_intent"] == {
        "requested_intent": "relationship",
        "resolved_intent": "general_hybrid",
        "intent_source": "fallback",
        "fallback_reason": "no_traversable_entity",
    }


@pytest.mark.asyncio
async def test_search_rejects_unknown_intent_before_adapter_access():
    engine = SearchEngine(
        relational=object(),
        keyword=object(),
        vector=object(),
        embed_cfg={},
        config=RetrievalConfig(),
    )

    with pytest.raises(ValueError, match="unsupported retrieval intent"):
        await engine.search("payroll", intent="semantic_lookup")


@pytest.mark.asyncio
async def test_search_path_uses_entity_linker_not_legacy_query_analysis(db, monkeypatch):
    active = _memory("m-active", "PostgreSQL pooling memory")
    await db.insert_memory(active)
    entity_id = await db.upsert_entity("postgresql", "PostgreSQL")
    await db.link_memory_entity(active.id, entity_id)

    adapters = build_sqlite_adapters(db, FakeCollection([active.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]
    assert not hasattr(engine, "_build_known_entities")

    result = await engine.search("PostgreSQL", entities=["postgresql"], top_k=10)

    assert [r.memory_id for r in result["results"]] == [active.id]
    assert result["query_analysis"]["detected_entities"] == ["postgresql"]
    assert result["query_analysis"]["entity_linking"][0]["channel"] == "explicit"


@pytest.mark.asyncio
async def test_search_recalls_memory_from_source_title_metadata(db, monkeypatch):
    target = _memory("m-blocker", "Lifecycle assignment skips person assignment creation")
    await db.insert_memory(target)
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _document(
        db,
        "SFPAY-179397",
        "src-jira",
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await db.add_memory_source("m-blocker", "SFPAY-179397", "jira", None, source_updated_at=None)

    adapters = build_sqlite_adapters(db, FakeCollection([]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("create blocker hint", top_k=10)

    assert [r.memory_id for r in result["results"]] == ["m-blocker"]
    assert "bm25_metadata_tokens" in result["query_analysis"]["strategies_used"]
    evidence = result["results"][0].retrieval_evidence
    assert evidence is not None
    assert evidence["metadata_lexical"] == {
        "channel": "bm25_metadata_tokens",
        "query_paths": ["full_query"],
        "matched_fields": ["metadata_any"],
        "matched_text": [
            "SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment | "
            "SFPAY-179397 | PAY | MountTai Defects | https://x/SFPAY-179397"
        ],
        "source_refs": [
            {
                "source_id": "src-jira",
                "doc_id": "SFPAY-179397",
                "source_type": "jira",
            }
        ],
    }
    assert evidence["rank_fusion"]["intent"] == "general_hybrid"
    assert evidence["rank_fusion"]["contributions"] == [
        {
            "channel": "metadata_lexical",
            "rank": 1,
            "channel_weight": 0.15,
            "multiplier": 1.0,
            "weighted_score": pytest.approx(0.15 / (DEFAULT_RRF_K + 1)),
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        'Find the memory titled "Create Access Review in Quarterly Payroll"',
        "查找标题为“Create Access Review in Quarterly Payroll”的 memory",
    ],
)
async def test_requested_known_item_recalls_quoted_title_inside_natural_language_query(
    db,
    query,
):
    target, engine = await _title_only_search_engine(db)

    result = await engine.search(
        query,
        intent="known_item",
        top_k=10,
    )

    assert result["retrieval_intent"] == {
        "requested_intent": "known_item",
        "resolved_intent": "known_item",
        "intent_source": "requested",
        "fallback_reason": None,
    }
    assert [item.memory_id for item in result["results"]] == [target.id]
    evidence = result["results"][0].retrieval_evidence
    assert evidence is not None
    assert evidence["metadata_lexical"]["channel"] == "bm25_metadata_tokens"
    assert evidence["metadata_lexical"]["query_paths"] == ["quoted_identity"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ('Find ""', None),
        ('Find "First title" and "Second title"', "First title"),
        ('Find "   " then "Second title"', "Second title"),
        (f'Find "{"x" * 257}" then "Bounded title"', "Bounded title"),
        ('Find "line\nbreak"', None),
        ('查找“First   title”或“Second title”', "First title"),
    ],
)
def test_quoted_identity_query_selects_first_non_empty_bounded_single_line_span(
    query,
    expected,
):
    assert _quoted_identity_query(query) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("full_score", "quoted_score", "expected_paths"),
    [
        (1.0, 0.5, ("full_query",)),
        (0.5, 1.0, ("quoted_identity",)),
        (1.0, 1.0, ("full_query", "quoted_identity")),
    ],
)
async def test_metadata_query_paths_only_report_retained_channel_hits(
    full_score,
    quoted_score,
    expected_paths,
):
    engine = SearchEngine(
        relational=object(),
        keyword=QueryScoredKeyword(
            full_score=full_score,
            quoted_score=quoted_score,
        ),
        vector=object(),
        embed_cfg={},
        config=RetrievalConfig(),
    )

    hits, paths = await engine._metadata_searches(
        'Find the memory titled "Create Access Review in Quarterly Payroll"',
        "known_item",
        None,
        object(),
        10,
        source_filter=None,
        time_range=None,
    )

    assert [hit.score for hit in hits] == [max(full_score, quoted_score)]
    assert paths == {"m-access-review": expected_paths}


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", [None, "general_hybrid"])
async def test_quoted_title_is_not_expanded_without_requested_known_item(
    db,
    intent,
):
    _target, engine = await _title_only_search_engine(db)

    result = await engine.search(
        'Find the memory titled "Create Access Review in Quarterly Payroll"',
        intent=intent,
        top_k=10,
    )

    assert result["results"] == []


@pytest.mark.asyncio
async def test_search_recalls_compound_query_from_metadata_trigram(db, monkeypatch):
    target = _memory("m-blocker", "Lifecycle assignment skips person assignment creation")
    await db.insert_memory(target)
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _document(
        db,
        "SFPAY-179397",
        "src-jira",
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await db.add_memory_source("m-blocker", "SFPAY-179397", "jira", None, source_updated_at=None)

    adapters = build_sqlite_adapters(db, FakeCollection([]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("create blockerhints", top_k=10)

    assert [r.memory_id for r in result["results"]] == ["m-blocker"]
    evidence = result["results"][0].retrieval_evidence
    assert evidence is not None
    assert evidence["metadata_lexical"]["channel"] == "metadata_trigram"
    assert "metadata_trigram" in evidence["metadata_lexical"]["matched_fields"]


@pytest.mark.asyncio
async def test_source_filter_prevents_non_matching_metadata_evidence(db, monkeypatch):
    shared = _memory("m-shared", "Lifecycle assignment skips person assignment creation")
    await db.insert_memory(shared)
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await db.upsert_source(
        "src-wiki", "confluence", "Payroll Wiki", "{}", access_policy="workspace", owner_user_id="dev"
    )
    await _document(
        db,
        "SFPAY-179397",
        "src-jira",
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await _document(
        db,
        "wiki-runbook",
        "src-wiki",
        title="Payroll lifecycle runbook",
    )
    await db.add_memory_source("m-shared", "SFPAY-179397", "jira", None, source_updated_at=None)
    await db.add_memory_source("m-shared", "wiki-runbook", "confluence", None, source_updated_at=None)

    adapters = build_sqlite_adapters(db, FakeCollection(["m-shared"]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search(
        "create blocker hint",
        source_filter=MemorySourceFilter(source_ids=("src-wiki",)),
        top_k=10,
    )

    assert [r.memory_id for r in result["results"]] == ["m-shared"]
    evidence = result["results"][0].retrieval_evidence or {}
    assert "metadata_lexical" not in evidence


@pytest.mark.asyncio
async def test_rerank_prompt_includes_metadata_evidence_for_metadata_hits(db, monkeypatch):
    target = _memory("m-blocker", "Lifecycle assignment skips person assignment creation")
    await db.insert_memory(target)
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _document(
        db,
        "SFPAY-179397",
        "src-jira",
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await db.add_memory_source("m-blocker", "SFPAY-179397", "jira", None, source_updated_at=None)

    reranker = RecordingRerankClient()
    adapters = build_sqlite_adapters(db, FakeCollection([]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(enable_reranking=True, rerank_candidates=10),
        structured_llm_client=reranker,
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("create blocker hint", top_k=10)

    assert [r.memory_id for r in result["results"]] == ["m-blocker"]
    assert reranker.prompt is not None
    assert "Retrieval evidence:" in reranker.prompt
    assert "Create Blocker Hint in On Demand Lifecycle Assignment" in reranker.prompt
    assert "jira:SFPAY-179397" in reranker.prompt


@pytest.mark.asyncio
async def test_source_filter_applies_to_vector_hits(db, monkeypatch):
    # Both memories are surfaced by the vector channel (and BM25, since both
    # match the FTS query); only m-backed is supported by a document from
    # source "wiki". The fused-set source filter must drop m-unbacked, so a
    # hit cannot bypass the filter by riding the vector channel.
    backed = _memory("m-backed", "PostgreSQL pooling from the wiki")
    unbacked = _memory("m-unbacked", "PostgreSQL pooling from elsewhere")
    await db.insert_memory(backed)
    await db.insert_memory(unbacked)
    await _document(db, "doc-wiki", "wiki")
    await _document(db, "doc-other", "other")
    await db.add_memory_source("m-backed", "doc-wiki", "wiki", None, source_updated_at=None)
    await db.add_memory_source("m-unbacked", "doc-other", "other", None, source_updated_at=None)

    adapters = build_sqlite_adapters(db, FakeCollection(["m-backed", "m-unbacked"]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("PostgreSQL", source_filter=MemorySourceFilter(source_ids=("wiki",)), top_k=10)
    assert [r.memory_id for r in result["results"]] == ["m-backed"]


@pytest.mark.asyncio
async def test_structured_source_filter_accepts_multiple_source_ids(db, monkeypatch):
    from_structured_filter = _memory("m-structured", "PostgreSQL pooling from wiki")
    from_top_level_sources = _memory("m-top-level", "PostgreSQL pooling from Jira")
    filtered_out = _memory("m-other", "PostgreSQL pooling from Slack")
    await db.insert_memory(from_structured_filter)
    await db.insert_memory(from_top_level_sources)
    await db.insert_memory(filtered_out)
    await _document(db, "doc-wiki", "wiki")
    await _document(db, "doc-jira", "jira")
    await _document(db, "doc-slack", "slack")
    await db.add_memory_source("m-structured", "doc-wiki", "confluence", None, source_updated_at=None)
    await db.add_memory_source("m-top-level", "doc-jira", "jira", None, source_updated_at=None)
    await db.add_memory_source("m-other", "doc-slack", "slack", None, source_updated_at=None)

    adapters = build_sqlite_adapters(
        db,
        FakeCollection(["m-structured", "m-top-level", "m-other"]),
    )
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search(
        "PostgreSQL",
        source_filter=MemorySourceFilter(source_ids=("wiki", "jira")),
        top_k=10,
    )

    assert [r.memory_id for r in result["results"]] == ["m-structured", "m-top-level"]


@pytest.mark.asyncio
async def test_structured_source_filter_applies_to_vector_hits(db, monkeypatch):
    codex = _memory(
        "m-codex",
        "Scheduler claim was patched by Codex",
        repo_identifier="github.tools.sap/hcm/memforge-cloud",
    )
    jira = _memory("m-jira", "Scheduler issue from Jira")
    other_repo = _memory(
        "m-other-repo",
        "Scheduler claim was patched elsewhere",
        repo_identifier="github.tools.sap/hcm/other",
    )
    await db.insert_memory(codex)
    await db.insert_memory(jira)
    await db.insert_memory(other_repo)
    await _document(db, "doc-codex", "src-agent-codex", client="codex")
    await _document(db, "doc-jira", "src-jira")
    await _document(db, "doc-other-repo", "src-agent-codex", client="codex")
    await db.add_memory_source("m-codex", "doc-codex", "agent_session", None, source_updated_at=None)
    await db.add_memory_source("m-jira", "doc-jira", "jira", None, source_updated_at=None)
    await db.add_memory_source(
        "m-other-repo",
        "doc-other-repo",
        "agent_session",
        None,
        source_updated_at=None,
    )

    adapters = build_sqlite_adapters(
        db,
        FakeCollection(["m-codex", "m-jira", "m-other-repo"]),
    )
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search(
        "Scheduler claim",
        source_filter=MemorySourceFilter(
            clients=("codex",),
            repo_identifiers=("github.tools.sap/hcm/memforge-cloud",),
        ),
        top_k=10,
    )

    assert [r.memory_id for r in result["results"]] == ["m-codex"]


@pytest.mark.asyncio
async def test_explicit_time_range_filters_vector_hits_before_ranking(db, monkeypatch):
    in_window = _memory("m-in-window", "Payroll incident triage pattern")
    out_of_window = _memory("m-out-of-window", "Payroll incident triage pattern")
    await db.insert_memory(in_window)
    await db.insert_memory(out_of_window)
    await _document(db, "doc-fresh", "wiki")
    await _document(db, "doc-stale", "wiki")
    await db.add_memory_source(
        "m-in-window",
        "doc-fresh",
        "confluence",
        None,
        source_updated_at=datetime(2026, 6, 20, tzinfo=timezone.utc),
    )
    await db.add_memory_source(
        "m-out-of-window",
        "doc-stale",
        "confluence",
        None,
        source_updated_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )

    adapters = build_sqlite_adapters(
        db,
        FakeCollection(["m-out-of-window", "m-in-window"]),
    )
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search(
        "Payroll incident",
        source_filter=MemorySourceFilter(source_ids=("wiki",)),
        time_range=MemoryTimeRange(
            after=datetime(2026, 6, 19, tzinfo=timezone.utc),
            before=datetime(2026, 6, 21, tzinfo=timezone.utc),
            date_type="source_updated_at",
        ),
        top_k=10,
    )

    assert [r.memory_id for r in result["results"]] == ["m-in-window"]


@pytest.mark.asyncio
async def test_known_item_prefers_rank_one_metadata_over_vector_plus_content(db, monkeypatch):
    metadata_target = _memory("m-metadata", "Durable source-backed note with different wording")
    content_competitor = _memory("m-content", "ISSUE 12345 lifecycle blocker note from content")
    await db.insert_memory(metadata_target)
    await db.insert_memory(content_competitor)
    await db.upsert_source("src-issues", "issue", "Issue Tracker", "{}", access_policy="workspace", owner_user_id="dev")
    await _document(
        db,
        "ISSUE-12345",
        "src-issues",
        title="ISSUE-12345 lifecycle blocker",
    )
    await db.add_memory_source("m-metadata", "ISSUE-12345", "issue", None, source_updated_at=None)

    adapters = build_sqlite_adapters(db, FakeCollection(["m-content", "m-metadata"]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("ISSUE-12345 lifecycle blocker", top_k=2)

    assert result["retrieval_intent"]["resolved_intent"] == "known_item"
    assert [r.memory_id for r in result["results"]][:2] == ["m-metadata", "m-content"]
    assert result["results"][0].retrieval_evidence["metadata_lexical"]["channel"] == "bm25_metadata_tokens"


@pytest.mark.asyncio
async def test_code_symbol_metadata_hit_resolves_known_item(db, monkeypatch):
    metadata_target = _memory("m-code-symbol", "Implementation note with durable source support")
    semantic_competitor = _memory("m-semantic", "Payroll cutoff command retries after lock contention")
    await db.insert_memory(metadata_target)
    await db.insert_memory(semantic_competitor)
    await db.upsert_source(
        "src-code", "github", "Payroll Repository", "{}", access_policy="workspace", owner_user_id="dev"
    )
    await _document(
        db,
        "src/payroll/PayrollCutoffCommand.py",
        "src-code",
        title="PayrollCutoffCommand.resolveWindow source reference",
    )
    await db.add_memory_source(
        "m-code-symbol",
        "src/payroll/PayrollCutoffCommand.py",
        "github",
        None,
        source_updated_at=None,
    )

    adapters = build_sqlite_adapters(db, FakeCollection(["m-semantic", "m-code-symbol"]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("PayrollCutoffCommand", top_k=2)

    assert result["retrieval_intent"]["resolved_intent"] == "known_item"
    assert result["results"][0].memory_id == "m-code-symbol"


@pytest.mark.asyncio
async def test_code_symbol_without_metadata_support_resolves_known_item(db, monkeypatch):
    metadata_target = _memory("m-metadata", "Blocker hint Jira task summary")
    semantic_competitor = _memory("m-semantic", "Command implementation retry analysis")
    await db.insert_memory(metadata_target)
    await db.insert_memory(semantic_competitor)
    await db.upsert_source("src-jira", "jira", "Mount Tai Jira", "{}", access_policy="workspace", owner_user_id="dev")
    await _document(
        db,
        "SFPAY-100",
        "src-jira",
        title="Create blocker hint lifecycle task",
    )
    await db.add_memory_source("m-metadata", "SFPAY-100", "jira", None, source_updated_at=None)

    adapters = build_sqlite_adapters(db, FakeCollection(["m-semantic", "m-metadata"]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    async def metadata_results(
        _query,
        _memory_types,
        _scope,
        limit,
        *,
        source_filter,
        time_range,
    ):
        return [
            KeywordCandidate(
                memory_id="m-metadata",
                score=10.0,
                channel="bm25_metadata_tokens",
                matched_text=("Create blocker hint lifecycle task",),
            )
        ]

    monkeypatch.setattr(engine, "_bm25_metadata_search", metadata_results)

    result = await engine.search("PersistCutOffBlockerHintsCommand blocker hint", top_k=2)

    assert result["retrieval_intent"]["resolved_intent"] == "known_item"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "matched_title"),
    [
        ("edit a rule locked by another user", "Edit a rule locked by another user"),
        ("编辑一个被其他用户锁定的规则", "编辑一个被其他用户锁定的规则"),
    ],
)
async def test_exact_metadata_channel_evidence_resolves_known_item_independent_of_language(
    db,
    monkeypatch,
    query,
    matched_title,
):
    metadata_target = _memory("m-metadata", "Durable source-backed note")
    await db.insert_memory(metadata_target)
    adapters = build_sqlite_adapters(db, FakeCollection([metadata_target.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda value: [0.1]

    async def metadata_results(
        _query,
        _memory_types,
        _scope,
        limit,
        *,
        source_filter,
        time_range,
    ):
        return [
            KeywordCandidate(
                memory_id=metadata_target.id,
                score=10.0,
                channel="bm25_metadata_tokens",
                matched_text=(matched_title,),
            )
        ]

    monkeypatch.setattr(engine, "_bm25_metadata_search", metadata_results)

    result = await engine.search(query, top_k=1)

    assert result["retrieval_intent"]["resolved_intent"] == "known_item"
    assert result["retrieval_intent"]["intent_source"] == "deterministic"


@pytest.mark.asyncio
async def test_generic_metadata_token_hit_does_not_force_known_item(db, monkeypatch):
    metadata_target = _memory("m-metadata", "Durable source-backed note")
    await db.insert_memory(metadata_target)
    adapters = build_sqlite_adapters(db, FakeCollection([metadata_target.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda value: [0.1]

    async def metadata_results(*args, **kwargs):
        return [
            KeywordCandidate(
                memory_id=metadata_target.id,
                score=10.0,
                channel="bm25_metadata_tokens",
                matched_text=("Payroll Jira | Quarterly payroll review",),
            )
        ]

    monkeypatch.setattr(engine, "_bm25_metadata_search", metadata_results)

    result = await engine.search("payroll", top_k=1)

    assert result["retrieval_intent"]["resolved_intent"] == "general_hybrid"
    assert result["retrieval_intent"]["intent_source"] == "default"


@pytest.mark.asyncio
async def test_metadata_trigram_hit_does_not_force_known_item(db, monkeypatch):
    metadata_target = _memory("m-metadata", "Durable source-backed note")
    await db.insert_memory(metadata_target)

    adapters = build_sqlite_adapters(db, FakeCollection(["m-metadata"]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    async def metadata_results(
        _query,
        _memory_types,
        _scope,
        limit,
        *,
        source_filter,
        time_range,
    ):
        return [
            KeywordCandidate(
                memory_id="m-metadata",
                score=10.0,
                channel="metadata_trigram",
                matched_text=("Create Blocker Hint in On Demand Lifecycle Assignment",),
            )
        ]

    monkeypatch.setattr(engine, "_bm25_metadata_search", metadata_results)

    result = await engine.search("create blocker hint", top_k=1)

    assert result["retrieval_intent"]["resolved_intent"] == "general_hybrid"


@pytest.mark.asyncio
async def test_explanatory_query_defaults_to_general_hybrid(db, monkeypatch):
    memory = _memory("m-semantic", "Payment retry policy after payroll cutoff")
    await db.insert_memory(memory)

    adapters = build_sqlite_adapters(db, FakeCollection([memory.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search(
        "why does payroll cutoff affect payment retry policy",
        top_k=1,
    )

    assert result["retrieval_intent"]["resolved_intent"] == "general_hybrid"


@pytest.mark.asyncio
async def test_requested_relationship_downweights_broad_entity_fanout(db, monkeypatch):
    specific = _memory("m-specific-target", "Specific graph target")
    broad_target = _memory("m-broad-target", "Broad graph target")
    await db.insert_memory(specific)
    await db.insert_memory(broad_target)
    specific_entity = await db.upsert_entity("specific topic", "Specific Topic")
    broad_entity = await db.upsert_entity("broad topic", "Broad Topic")
    await db.link_memory_entity(specific.id, specific_entity)
    await db.link_memory_entity(broad_target.id, broad_entity)
    for index in range(100):
        memory = _memory(f"m-broad-{index:03d}", f"Broad related memory {index}")
        await db.insert_memory(memory)
        await db.link_memory_entity(memory.id, broad_entity)

    adapters = build_sqlite_adapters(db, FakeCollection([]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    async def empty_channel(*args, **kwargs):
        return []

    monkeypatch.setattr(engine, "_vector_search", empty_channel)
    monkeypatch.setattr(engine, "_bm25_search", empty_channel)
    monkeypatch.setattr(engine, "_bm25_metadata_search", empty_channel)

    result = await engine.search(
        "related risks dependencies around",
        entities=["specific topic", "broad topic"],
        intent="relationship",
        top_k=5,
    )

    assert result["retrieval_intent"]["resolved_intent"] == "relationship"
    linked = {item["canonical_name"]: item for item in result["query_analysis"]["entity_linking"]}
    assert linked["specific topic"]["specificity"] == pytest.approx(1.0)
    assert linked["broad topic"]["visible_memory_count"] >= 100
    assert linked["broad topic"]["specificity"] < 1.0
    assert result["results"][0].memory_id == specific.id
    graph_evidence = result["results"][0].retrieval_evidence or {}
    assert graph_evidence["rank_fusion"]["intent"] == "relationship"
    assert graph_evidence["rank_fusion"]["contributions"] == [
        {
            "channel": "graph",
            "rank": 1,
            "channel_weight": 0.3,
            "multiplier": 1.0,
            "weighted_score": pytest.approx(0.3 / (DEFAULT_RRF_K + 1)),
        }
    ]


def test_graph_contributions_choose_best_specific_entity_without_summing() -> None:
    broad = EntityLinkCandidate(
        entity_id=1,
        canonical_name="Broad Topic",
        matched_alias="Broad Topic",
        channel="explicit",
        contributing_channels=("explicit",),
        score=1.0,
        matched_text="Broad Topic",
        activates_graph=True,
        visible_memory_count=250,
        visible_source_count=20,
        specificity=0.3,
    )
    specific = EntityLinkCandidate(
        entity_id=2,
        canonical_name="Specific Topic",
        matched_alias="Specific Topic",
        channel="explicit",
        contributing_channels=("explicit",),
        score=1.0,
        matched_text="Specific Topic",
        activates_graph=True,
        visible_memory_count=1,
        visible_source_count=1,
        specificity=1.0,
    )

    contributions = SearchEngine._graph_contributions(
        [broad, specific],
        [[("m-shared", 1.0)], [("m-shared", 1.0)]],
    )

    assert contributions["m-shared"].entity_id == specific.entity_id
    assert contributions["m-shared"].multiplier == 1.0


@pytest.mark.asyncio
async def test_queried_search_honors_offset_after_ranking(db, monkeypatch):
    first = _memory("m-first", "PostgreSQL pagination memory first")
    second = _memory("m-second", "PostgreSQL pagination memory second")
    third = _memory("m-third", "PostgreSQL pagination memory third")
    for memory in (first, second, third):
        await db.insert_memory(memory)

    async def no_bm25(*args, **kwargs):
        return []

    adapters = build_sqlite_adapters(
        db,
        FakeCollection(["m-first", "m-second", "m-third"]),
    )
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]
    monkeypatch.setattr(engine, "_bm25_search", no_bm25)

    result = await engine.search("PostgreSQL pagination", top_k=1, offset=1)

    assert [r.memory_id for r in result["results"]] == ["m-second"]
    assert result["total_candidates"] == 3
    assert "total_count" not in result


@pytest.mark.asyncio
async def test_queried_search_top_k_does_not_change_ranking_prefix(db, monkeypatch):
    memories = [_memory(f"m-{index}", f"Payroll rank window memory {index}") for index in range(40)]
    for memory in memories:
        await db.insert_memory(memory)

    vector_order = [memory.id for memory in memories]
    adapters = build_sqlite_adapters(db, FakeCollection(vector_order))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    lexical_order = [f"m-{index}" for index in range(20, 40)] + [f"m-{index}" for index in range(20)]

    async def bm25_results(_query, _analysis, _memory_types, _scope, limit):
        return [(memory_id, float(len(lexical_order) - index)) for index, memory_id in enumerate(lexical_order[:limit])]

    async def metadata_results(
        _query,
        _memory_types,
        _scope,
        limit,
        *,
        source_filter,
        time_range,
    ):
        return [
            KeywordCandidate(
                memory_id=memory_id,
                score=float(len(lexical_order) - index),
                channel="bm25_metadata_tokens",
            )
            for index, memory_id in enumerate(lexical_order[:limit])
        ]

    monkeypatch.setattr(engine, "_bm25_search", bm25_results)
    monkeypatch.setattr(engine, "_bm25_metadata_search", metadata_results)

    first_page = await engine.search("Payroll rank window", top_k=10)
    larger_page = await engine.search("Payroll rank window", top_k=20)

    assert [r.memory_id for r in first_page["results"]] == [r.memory_id for r in larger_page["results"][:10]]
    assert first_page["ranking_window_size"] == larger_page["ranking_window_size"]
    assert first_page["candidate_count_kind"] == "windowed"
    assert first_page["has_more"] is True


@pytest.mark.asyncio
async def test_queried_search_top_k_prefix_stable_with_sqlite_metadata_channel(db, monkeypatch):
    memories = [_memory(f"m-sqlite-{index}", f"Unrelated memory body {index}") for index in range(40)]
    for index, memory in enumerate(memories):
        await db.insert_memory(memory)
        await _document(
            db,
            f"doc-sqlite-{index}",
            "src-jira",
            title=f"Create Blocker Hint source title {index:02d}",
        )
        await db.add_memory_source(
            memory.id,
            f"doc-sqlite-{index}",
            "jira",
            support_kind="extracted",
            source_updated_at=None,
        )

    vector_order = [memory.id for memory in reversed(memories)]
    adapters = build_sqlite_adapters(db, FakeCollection(vector_order))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    first_page = await engine.search("create blocker hint", top_k=10)
    larger_page = await engine.search("create blocker hint", top_k=20)

    assert [r.memory_id for r in first_page["results"]] == [r.memory_id for r in larger_page["results"][:10]]
    assert first_page["ranking_window_size"] == larger_page["ranking_window_size"] == DEFAULT_RANK_WINDOW_SIZE
    assert "bm25_metadata_tokens" in first_page["query_analysis"]["strategies_used"]
    assert any(
        result.retrieval_evidence
        and result.retrieval_evidence.get("metadata_lexical", {}).get("channel") == "bm25_metadata_tokens"
        for result in first_page["results"]
    )


@pytest.mark.asyncio
async def test_queried_search_legacy_config_uses_default_rank_window(db, monkeypatch):
    memory = _memory("m-legacy", "Payroll rank window memory")
    await db.insert_memory(memory)

    adapters = build_sqlite_adapters(db, FakeCollection([memory.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=SimpleNamespace(
            embedding_cache_size=256,
            rrf_k=60,
            recency_half_life_days=90,
            enable_reranking=False,
        ),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]
    seen_limits = []

    async def vector_results(_query, _memory_types, _scope, limit):
        seen_limits.append(limit)
        return [(memory.id, 1.0)]

    async def bm25_results(_query, _analysis, _memory_types, _scope, limit):
        seen_limits.append(limit)
        return []

    async def metadata_results(
        _query,
        _memory_types,
        _scope,
        limit,
        *,
        source_filter,
        time_range,
    ):
        seen_limits.append(limit)
        return []

    monkeypatch.setattr(engine, "_vector_search", vector_results)
    monkeypatch.setattr(engine, "_bm25_search", bm25_results)
    monkeypatch.setattr(engine, "_bm25_metadata_search", metadata_results)

    result = await engine.search("Payroll rank window", top_k=10)

    assert seen_limits == [DEFAULT_RANK_WINDOW_SIZE] * 3
    assert result["ranking_window_size"] == DEFAULT_RANK_WINDOW_SIZE


@pytest.mark.asyncio
async def test_search_engine_returns_only_memory_results_even_when_top_k_has_room(db, monkeypatch):
    memory = _memory("m-target", "Mount Tai payroll defect memory")
    await db.insert_memory(memory)
    await _document(db, "doc-target", "src-target")
    await db.add_memory_source(
        "m-target",
        "doc-target",
        "jira",
        None,
        source_updated_at=datetime(2026, 6, 25, tzinfo=timezone.utc),
    )

    async def no_bm25(*args, **kwargs):
        return []

    memory_adapters = build_sqlite_adapters(db, FakeCollection(["m-target"]))
    engine = SearchEngine(
        relational=memory_adapters.relational,
        keyword=memory_adapters.keyword,
        vector=memory_adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]
    monkeypatch.setattr(engine, "_bm25_search", no_bm25)

    result = await engine.search(
        "Mount Tai payroll defect",
        source_filter=MemorySourceFilter(source_ids=("src-target",)),
        top_k=2,
    )

    assert [r.memory_id for r in result["results"]] == ["m-target"]
    assert all(r.memory_id is not None for r in result["results"])


@pytest.mark.asyncio
async def test_queryless_source_id_time_range_uses_relational_listing_only(db, monkeypatch):
    newer = _memory("m-newer", "Mount Tai defect triage rule")
    older = _memory("m-older", "Mount Tai payroll defect rule")
    other = _memory("m-other", "Another source rule")
    await db.insert_memory(newer)
    await db.insert_memory(older)
    await db.insert_memory(other)
    await _document(db, "doc-newer", "src-mounttai")
    await _document(db, "doc-older", "src-mounttai")
    await _document(db, "doc-other", "src-other")
    await db.add_memory_source(
        "m-newer",
        "doc-newer",
        "jira",
        None,
        source_updated_at=datetime(2026, 6, 25, tzinfo=timezone.utc),
    )
    await db.add_memory_source(
        "m-older",
        "doc-older",
        "jira",
        None,
        source_updated_at=datetime(2026, 6, 24, tzinfo=timezone.utc),
    )
    await db.add_memory_source(
        "m-other",
        "doc-other",
        "jira",
        None,
        source_updated_at=datetime(2026, 6, 26, tzinfo=timezone.utc),
    )

    adapters = build_sqlite_adapters(
        db,
        FakeCollection(["m-other", "m-older", "m-newer"]),
    )
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: (_ for _ in ()).throw(
        AssertionError("queryless search must not embed")
    )

    result = await engine.search(
        "",
        source_filter=MemorySourceFilter(source_ids=("src-mounttai",)),
        time_range=MemoryTimeRange(
            after=datetime(2026, 6, 20, tzinfo=timezone.utc),
            before=datetime(2026, 6, 27, tzinfo=timezone.utc),
            date_type="source_updated_at",
        ),
        top_k=10,
    )

    assert [r.memory_id for r in result["results"]] == ["m-newer", "m-older"]
    assert result["total_candidates"] == 2
    assert result["query_analysis"]["strategies_used"] == ["source_time_listing"]


@pytest.mark.asyncio
async def test_queryless_search_compatibility_applies_memory_types_before_pagination(db):
    decision = _memory("m-queryless-decision", "Decision", memory_type="decision")
    fact = _memory("m-queryless-fact", "Fact")
    for memory in (decision, fact):
        await db.insert_memory(memory)
        await _document(db, f"doc-{memory.id}", "src-jira")
        await db.add_memory_source(
            memory.id,
            f"doc-{memory.id}",
            "jira",
            None,
            source_updated_at=datetime(2026, 7, 25, tzinfo=timezone.utc),
        )
    adapters = build_sqlite_adapters(db, FakeCollection([]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )

    result = await engine.search(
        "",
        memory_types=["decision"],
        source_filter=MemorySourceFilter(source_ids=("src-jira",)),
        time_range=MemoryTimeRange(
            after=datetime(2026, 7, 20, tzinfo=timezone.utc),
            before=datetime(2026, 7, 28, tzinfo=timezone.utc),
            date_type="source_updated_at",
        ),
        top_k=1,
    )

    assert [item.memory_id for item in result["results"]] == ["m-queryless-decision"]
    assert result["total_candidates"] == 1


@pytest.mark.asyncio
async def test_recent_memory_listing_pages_with_a_request_bound_keyset_cursor(db):
    newer = _memory("m-newer-recent", "Newer decision", memory_type="decision")
    older = _memory("m-older-recent", "Older decision", memory_type="decision")
    excluded_type = _memory("m-fact-recent", "Fact in the same source")
    for memory in (newer, older, excluded_type):
        await db.insert_memory(memory)
        await _document(db, f"doc-{memory.id}", "src-jira")
    await db.add_memory_source(
        newer.id,
        f"doc-{newer.id}",
        "jira",
        None,
        source_updated_at=datetime(2026, 7, 25, tzinfo=timezone.utc),
    )
    await db.add_memory_source(
        older.id,
        f"doc-{older.id}",
        "jira",
        None,
        source_updated_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )
    await db.add_memory_source(
        excluded_type.id,
        f"doc-{excluded_type.id}",
        "jira",
        None,
        source_updated_at=datetime(2026, 7, 26, tzinfo=timezone.utc),
    )

    adapters = build_sqlite_adapters(db, FakeCollection([]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    source_filter = MemorySourceFilter(source_ids=("src-jira",))
    window = MemoryTimeRange(
        after=datetime(2026, 7, 20, tzinfo=timezone.utc),
        before=datetime(2026, 7, 28, tzinfo=timezone.utc),
        date_type="source_updated_at",
    )

    first = await engine.list_recent_memories(
        source_filter=source_filter,
        time_range=window,
        memory_types=["decision"],
        page_size=1,
    )
    second = await engine.list_recent_memories(
        source_filter=source_filter,
        time_range=window,
        memory_types=["decision"],
        page_size=1,
        cursor=first["next_cursor"],
    )

    assert [result["memory_id"] for result in first["results"]] == ["m-newer-recent"]
    assert [result["memory_id"] for result in second["results"]] == ["m-older-recent"]
    assert first["results"][0]["matched_at"] == "2026-07-25T00:00:00+00:00"
    assert second["results"][0]["matched_at"] == "2026-07-24T00:00:00+00:00"
    assert "relevance_score" not in first["results"][0]
    assert first["result_kind"] == "current_memories"
    assert first["is_changelog"] is False
    assert first["candidate_count_kind"] == "exact"
    assert first["count_scope"] == "current_page_read"
    assert first["total_candidates"] == 2
    assert first["has_more"] is True
    assert first["next_cursor"]
    assert second["has_more"] is False
    assert second["next_cursor"] is None
    assert first["listing_watermark"] == second["listing_watermark"]
    assert first["cursor_kind"] == "keyset"
    assert first["consistency"] == "request_bound_watermark_not_mvcc_snapshot"
    assert "snapshot_watermark" not in first


@pytest.mark.asyncio
async def test_recent_memory_cursor_is_opaque_and_bound_to_the_original_filters(db):
    memory = _memory("m-cursor-bound", "Cursor-bound decision", memory_type="decision")
    second_memory = _memory("m-cursor-bound-2", "Second decision", memory_type="decision")
    for item, source_updated_at in (
        (memory, datetime(2026, 7, 25, tzinfo=timezone.utc)),
        (second_memory, datetime(2026, 7, 24, tzinfo=timezone.utc)),
    ):
        await db.insert_memory(item)
        await _document(db, f"doc-{item.id}", "src-jira")
        await db.add_memory_source(
            item.id,
            f"doc-{item.id}",
            "jira",
            None,
            source_updated_at=source_updated_at,
        )
    adapters = build_sqlite_adapters(db, FakeCollection([]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    source_filter = MemorySourceFilter(source_ids=("src-jira",))
    window = MemoryTimeRange(
        after=datetime(2026, 7, 20, tzinfo=timezone.utc),
        before=datetime(2026, 7, 28, tzinfo=timezone.utc),
        date_type="source_updated_at",
    )
    first = await engine.list_recent_memories(
        source_filter=source_filter,
        time_range=window,
        memory_types=["decision"],
        page_size=1,
    )
    cursor = first["next_cursor"]
    assert isinstance(cursor, str)

    with pytest.raises(ValueError, match="does not match"):
        await engine.list_recent_memories(
            source_filter=source_filter,
            time_range=window,
            memory_types=["fact"],
            page_size=1,
            cursor=cursor,
        )
    with pytest.raises(ValueError, match="invalid"):
        await engine.list_recent_memories(
            source_filter=source_filter,
            time_range=window,
            memory_types=["decision"],
            page_size=1,
            cursor="not-a-valid-cursor",
        )
    non_object_cursor = base64.urlsafe_b64encode(b"[]").rstrip(b"=").decode("ascii")
    with pytest.raises(ValueError, match="invalid"):
        await engine.list_recent_memories(
            source_filter=source_filter,
            time_range=window,
            memory_types=["decision"],
            page_size=1,
            cursor=non_object_cursor,
        )
