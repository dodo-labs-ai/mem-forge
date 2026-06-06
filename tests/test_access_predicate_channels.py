import pytest
from memforge.models import (
    Memory,
    Visibility,
    content_hash,
    SHARED_PROJECT_KEY,
    UNSORTED_PROJECT_KEY,
)
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.adapters.context import AccessScope


WORKSPACE = Visibility.WORKSPACE.value
PRIVATE = Visibility.PRIVATE.value


def _mem(mid: str, content: str, *, visibility=WORKSPACE, owner=None,
         project_key=SHARED_PROJECT_KEY, status="active") -> Memory:
    return Memory(
        id=mid,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content + mid),
        visibility=visibility,
        owner_user_id=owner,
        project_key=project_key,
        tags=[],
        status=status,
    )


def _scope(*, user_id="u-1", include_private=False) -> AccessScope:
    return AccessScope(
        user_id=user_id,
        open_projects=frozenset({SHARED_PROJECT_KEY}),
        member_projects=frozenset(),
        include_private=include_private,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="project-first",
    )


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "p.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_bm25_hides_other_users_private_memory(db):
    # Two memories with the same FTS-matchable token, one workspace, one U2 private.
    await db.insert_memory(_mem("m-shared", "deploy via argocd", visibility=WORKSPACE))
    await db.insert_memory(_mem("m-priv", "deploy via argocd",
                                 visibility=PRIVATE, owner="u-2"))
    adapters = build_sqlite_adapters(db, memory_collection=None)
    # TEAM search by U1: must not return U2's private memory.
    team = await adapters.keyword.search("argocd", _scope(user_id="u-1",
                                                          include_private=False),
                                          memory_types=None, limit=10)
    assert {mid for mid, _ in team} == {"m-shared"}
    # PERSONALIZED search by U1: still must not return U2's private memory.
    personalized = await adapters.keyword.search(
        "argocd", _scope(user_id="u-1", include_private=True),
        memory_types=None, limit=10,
    )
    assert {mid for mid, _ in personalized} == {"m-shared"}


@pytest.mark.asyncio
async def test_graph_hides_other_users_private_memory(db):
    e1_id = await db.upsert_entity("argocd", "tool")
    await db.insert_memory(_mem("g-shared", "deploys via argocd"))
    await db.link_memory_entity("g-shared", e1_id)
    await db.insert_memory(_mem("g-priv", "deploys via argocd",
                                 visibility=PRIVATE, owner="u-2"))
    await db.link_memory_entity("g-priv", e1_id)
    adapters = build_sqlite_adapters(db, memory_collection=None)
    hits = await adapters.relational.graph_search(
        [e1_id], _scope(user_id="u-1", include_private=False),
        memory_types=None, limit=10,
    )
    assert {mid for mid, _ in hits} == {"g-shared"}


@pytest.mark.asyncio
async def test_temporal_hides_other_users_private_memory(db):
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    await db.insert_memory(_mem("t-shared", "x"))
    await db.insert_memory(_mem("t-priv", "x", visibility=PRIVATE, owner="u-2"))
    adapters = build_sqlite_adapters(db, memory_collection=None)
    hits = await adapters.relational.temporal_search(
        after=now - timedelta(days=1), before=None,
        scope=_scope(user_id="u-1", include_private=False),
        memory_types=None, limit=10,
    )
    assert {mid for mid, _ in hits} == {"t-shared"}


class _Coll:
    """Minimal Chroma fake: stores items with metadata, supports a where-filter."""

    def __init__(self):
        self.items: dict[str, dict] = {}

    def upsert(self, *, ids, embeddings, metadatas, **_):
        for i, mid in enumerate(ids):
            self.items[mid] = {"embedding": embeddings[i],
                                "metadata": dict(metadatas[i])}

    def query(self, *, query_embeddings, n_results, where=None, **_):
        # Trivial cosine-equal match: return all ids that satisfy where.
        def matches(meta, clause):
            if "$and" in clause:
                return all(matches(meta, c) for c in clause["$and"])
            if "$or" in clause:
                return any(matches(meta, c) for c in clause["$or"])
            for k, v in clause.items():
                if isinstance(v, dict) and "$in" in v:
                    if meta.get(k) not in v["$in"]:
                        return False
                else:
                    if meta.get(k) != v:
                        return False
            return True

        ids, dists = [], []
        for mid, rec in self.items.items():
            if where is None or matches(rec["metadata"], where):
                ids.append(mid)
                dists.append(0.1)  # any monotonic distance
        return {"ids": [ids[:n_results]], "distances": [dists[:n_results]]}


@pytest.mark.asyncio
async def test_vector_hides_other_users_private_memory(db):
    from memforge.storage.adapters.sqlite.vector import SqliteVectorStore
    coll = _Coll()
    coll.upsert(
        ids=["v-shared"], embeddings=[[0.1, 0.1, 0.1]],
        metadatas=[{"status": "active", "visibility": WORKSPACE,
                    "owner_user_id": "", "project_key": SHARED_PROJECT_KEY,
                    "memory_type": "fact"}],
    )
    coll.upsert(
        ids=["v-priv"], embeddings=[[0.1, 0.1, 0.1]],
        metadatas=[{"status": "active", "visibility": PRIVATE,
                    "owner_user_id": "u-2", "project_key": SHARED_PROJECT_KEY,
                    "memory_type": "fact"}],
    )
    vs = SqliteVectorStore(coll)
    hits = await vs.query([0.1, 0.1, 0.1],
                          _scope(user_id="u-1", include_private=False),
                          memory_types=None, limit=10)
    assert {mid for mid, _ in hits} == {"v-shared"}


@pytest.mark.asyncio
async def test_vector_pre_filter_does_not_drop_dangling_project_workspace_memory(db):
    # The vector tier intentionally does not narrow by project_key (Chroma cannot
    # express the dangling-project fail-safe). A workspace memory whose project
    # key has no row in the projects table must still be a vector candidate;
    # the relational post-fusion re-check is the authority on project openness.
    from memforge.storage.adapters.sqlite.vector import SqliteVectorStore
    coll = _Coll()
    coll.upsert(
        ids=["v-dangling"], embeddings=[[0.1, 0.1, 0.1]],
        metadatas=[{"status": "active", "visibility": WORKSPACE,
                    "owner_user_id": "", "project_key": "DANGLING-X",
                    "memory_type": "fact"}],
    )
    vs = SqliteVectorStore(coll)
    hits = await vs.query([0.1, 0.1, 0.1],
                          _scope(user_id="u-1", include_private=False),
                          memory_types=None, limit=10)
    assert {mid for mid, _ in hits} == {"v-dangling"}


@pytest.mark.asyncio
async def test_search_engine_team_search_excludes_private(db, monkeypatch):
    # Reuses the construction shape from tests/test_search_engine_adapters.py:
    # build a SearchEngine over the adapters with a stubbed embedding.
    from memforge.retrieval.search import SearchEngine
    from memforge.config import RetrievalConfig
    from memforge.retrieval.query_analyzer import QueryAnalysis

    async def fake_analyze_query(*args, **kwargs):
        return QueryAnalysis()

    monkeypatch.setattr(
        "memforge.retrieval.search.analyze_query", fake_analyze_query
    )

    coll = _Coll()
    coll.upsert(
        ids=["se-shared"], embeddings=[[0.1, 0.1, 0.1]],
        metadatas=[{"status": "active", "visibility": WORKSPACE,
                    "owner_user_id": "", "project_key": SHARED_PROJECT_KEY,
                    "memory_type": "fact"}],
    )
    coll.upsert(
        ids=["se-priv"], embeddings=[[0.1, 0.1, 0.1]],
        metadatas=[{"status": "active", "visibility": PRIVATE,
                    "owner_user_id": "u-2", "project_key": SHARED_PROJECT_KEY,
                    "memory_type": "fact"}],
    )
    await db.insert_memory(_mem("se-shared", "team thing", visibility=WORKSPACE))
    await db.insert_memory(_mem("se-priv", "private thing",
                                 visibility=PRIVATE, owner="u-2"))
    adapters = build_sqlite_adapters(db, coll)
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1, 0.1, 0.1]

    team_scope = AccessScope(
        user_id="u-1",
        open_projects=frozenset({SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY}),
        member_projects=frozenset(),
        include_private=False,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="project-first",
    )
    result = await engine.search("thing", top_k=10, request_scope=team_scope)
    ids = {row.memory_id for row in result["results"]}
    assert "se-priv" not in ids
    assert "se-shared" in ids


@pytest.mark.asyncio
async def test_agent_hook_uses_personalized_predicate(db, monkeypatch):
    # U1 calls the hook with their own user_id; the hook must surface U1's own
    # private memories AND workspace memories, but NOT U2's private memories.
    from memforge.agent_hooks import (
        AgentHookContextRequest,
        build_agent_hook_context,
    )
    from memforge.retrieval.search import SearchEngine
    from memforge.config import RetrievalConfig
    from memforge.retrieval.query_analyzer import QueryAnalysis

    async def fake_analyze_query(*args, **kwargs):
        return QueryAnalysis()

    monkeypatch.setattr(
        "memforge.retrieval.search.analyze_query", fake_analyze_query
    )

    coll = _Coll()
    coll.upsert(
        ids=["h-shared"], embeddings=[[0.1, 0.1, 0.1]],
        metadatas=[{"status": "active", "visibility": WORKSPACE,
                    "owner_user_id": "", "project_key": SHARED_PROJECT_KEY,
                    "memory_type": "fact"}],
    )
    coll.upsert(
        ids=["h-u1-priv"], embeddings=[[0.1, 0.1, 0.1]],
        metadatas=[{"status": "active", "visibility": PRIVATE,
                    "owner_user_id": "u-1", "project_key": SHARED_PROJECT_KEY,
                    "memory_type": "fact"}],
    )
    coll.upsert(
        ids=["h-u2-priv"], embeddings=[[0.1, 0.1, 0.1]],
        metadatas=[{"status": "active", "visibility": PRIVATE,
                    "owner_user_id": "u-2", "project_key": SHARED_PROJECT_KEY,
                    "memory_type": "fact"}],
    )
    await db.insert_memory(_mem("h-shared", "deploy decision",
                                 visibility=WORKSPACE))
    await db.insert_memory(_mem("h-u1-priv", "deploy decision",
                                 visibility=PRIVATE, owner="u-1"))
    await db.insert_memory(_mem("h-u2-priv", "deploy decision",
                                 visibility=PRIVATE, owner="u-2"))

    adapters = build_sqlite_adapters(db, coll)
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1, 0.1, 0.1]

    request = AgentHookContextRequest(
        client="codex", hook="UserPromptSubmit",
        workspace="ws", repo=None, branch=None,
        prompt="why this deploy decision", touched_files=[],
        include_recent_changes=False, max_memories=10,
    )
    ctx = await build_agent_hook_context(
        db, request, principal_user_id="u-1", search_engine=engine,
    )
    ids = {row["id"] for row in ctx.get("memories", [])}
    assert "h-u2-priv" not in ids
    assert "h-shared" in ids
    assert "h-u1-priv" in ids


@pytest.mark.asyncio
async def test_agent_hook_recent_changes_excludes_other_users_private(db):
    # build_agent_hook_context also returns _recent_memory_changes when memory
    # context is enabled. It currently reads memories directly with only
    # status='active' and an optional repo filter, which would leak U2's private
    # rows into U1's hook context. The recent-changes path must apply the same
    # access predicate as _search_memories. The hook receives the principal
    # explicitly: a non-HTTP caller cannot fall back to body-derived identity.
    from memforge.agent_hooks import (
        AgentHookContextRequest,
        build_agent_hook_context,
    )

    await db.insert_memory(_mem("rc-shared", "team change", visibility=WORKSPACE))
    await db.insert_memory(_mem("rc-priv", "private change",
                                 visibility=PRIVATE, owner="u-2"))
    request = AgentHookContextRequest(
        client="codex", hook="SessionStart",
        workspace="ws", repo=None, branch=None,
        include_recent_changes=True, max_memories=5,
    )
    ctx = await build_agent_hook_context(db, request, principal_user_id="u-1")
    ids = {row["id"] for row in ctx.get("recent_changes", [])}
    assert "rc-priv" not in ids
    assert "rc-shared" in ids
