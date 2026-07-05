"""SearchEngine accepts adapters handles and routes channels through them."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.config import RetrievalConfig
from memforge.llm.structured import RerankResponse
from memforge.models import DocumentRecord, Memory, content_hash
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
from memforge.retrieval.search import SearchEngine
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


class FakeCollection:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def query(self, **kwargs):
        return {"ids": [self.ids], "distances": [[0.01 for _ in self.ids]]}

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


def _memory(
    mem_id: str,
    content: str,
    status: str = "active",
    repo_identifier: str | None = None,
) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
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


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "search-adapters.db"))
    await database.connect()
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
async def test_search_path_uses_entity_linker_not_legacy_query_analysis(db, monkeypatch):
    active = _memory("m-active", "PostgreSQL pooling memory")
    await db.insert_memory(active)
    entity_id = await db.upsert_entity("postgresql", "PostgreSQL", ["database"])
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
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}")
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
    assert evidence == {
        "metadata_lexical": {
            "channel": "bm25_metadata_tokens",
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
    }


@pytest.mark.asyncio
async def test_search_recalls_compound_query_from_metadata_trigram(db, monkeypatch):
    target = _memory("m-blocker", "Lifecycle assignment skips person assignment creation")
    await db.insert_memory(target)
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}")
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
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}")
    await db.upsert_source("src-wiki", "confluence", "Payroll Wiki", "{}")
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
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}")
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
