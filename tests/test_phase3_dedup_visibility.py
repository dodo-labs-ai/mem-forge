import pytest
from datetime import datetime, timezone

from memforge.models import (
    Memory,
    MemoryStatus,
    SHARED_PROJECT_KEY,
    Visibility,
    content_hash,
)
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.memory.store import MemoryStore

WORKSPACE = Visibility.WORKSPACE.value
PRIVATE = Visibility.PRIVATE.value
ACTIVE = MemoryStatus.ACTIVE.value


class _FakeColl:
    """Adversarial Chroma stub for the dedup write-side guard tests.

    `query()` returns every seeded id at distance 0.0, ignoring the `where`
    clause. The point is to bypass the access predicate's vector pre-filter
    so the test exercises the new guard in `deduplicate_and_insert`. A
    faithful Chroma fake would silently shield the test from a missing guard.
    """

    def __init__(self, *, seeded: list[str] | None = None) -> None:
        self._seeded = list(seeded or [])

    def upsert(self, *, ids, embeddings=None, metadatas=None, **_):
        for new_id in ids:
            if new_id not in self._seeded:
                self._seeded.append(new_id)

    def query(self, *, n_results=10, **_):
        ids = self._seeded[:n_results]
        return {"ids": [ids], "distances": [[0.0] * len(ids)]}

    def delete(self, *, ids):
        self._seeded = [i for i in self._seeded if i not in ids]


def _mem(mid, content, *, visibility=WORKSPACE, owner=None, project_key=SHARED_PROJECT_KEY, status=ACTIVE):
    return Memory(
        id=mid,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        visibility=visibility,
        owner_user_id=owner,
        project_key=project_key,
        tags=[],
        status=status,
    )


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "p3.db"))
    await database.connect()
    yield database
    await database.close()


async def _seed_doc(database: Database, doc_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await database.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, "src", f"http://test/{doc_id}", doc_id, "TEST", now, "1", f"h-{doc_id}", now),
    )
    await database.db.commit()


@pytest.mark.asyncio
async def test_private_write_does_not_corroborate_workspace(db, monkeypatch):
    # Seed a workspace memory that the vector channel will return as a dedup
    # candidate. The fake collection must include this id so the dedup loop
    # actually considers it; otherwise the test would pass for the wrong
    # reason (an empty candidate pool always returns "inserted").
    workspace = _mem("m-ws", "deploy via argocd", visibility=WORKSPACE)
    await db.insert_memory(workspace)

    # A private write with identical content must INSERT, not corroborate.
    adapters = build_sqlite_adapters(db, memory_collection=_FakeColl(seeded=["m-ws"]))
    store = MemoryStore(adapters.relational, adapters.keyword, adapters.vector, embed_cfg={}, dedup_threshold=0.08)

    async def _stub_embed(text):
        return [0.1, 0.1, 0.1]

    monkeypatch.setattr(store, "_embed", _stub_embed)

    private = _mem("m-priv", "deploy via argocd", visibility=PRIVATE, owner="u-alice")
    await _seed_doc(db, "d-1")
    result = await store.deduplicate_and_insert(
        private,
        doc_id="d-1",
        source_type="agent_session",
        source_updated_at=None,
    )
    assert result == "inserted"  # NOT "corroborated"


@pytest.mark.asyncio
async def test_workspace_write_does_not_corroborate_private(db, monkeypatch):
    # Seed an alice-private memory that the vector channel will return as a
    # dedup candidate (test it for real, not for an empty pool).
    private = _mem("m-priv", "deploy via argocd", visibility=PRIVATE, owner="u-alice")
    await db.insert_memory(private)

    # A workspace write with identical content must INSERT, not corroborate.
    adapters = build_sqlite_adapters(db, memory_collection=_FakeColl(seeded=["m-priv"]))
    store = MemoryStore(adapters.relational, adapters.keyword, adapters.vector, embed_cfg={}, dedup_threshold=0.08)

    async def _stub_embed(text):
        return [0.1, 0.1, 0.1]

    monkeypatch.setattr(store, "_embed", _stub_embed)

    workspace = _mem("m-ws", "deploy via argocd", visibility=WORKSPACE)
    await _seed_doc(db, "d-2")
    result = await store.deduplicate_and_insert(
        workspace,
        doc_id="d-2",
        source_type="confluence",
        source_updated_at=None,
    )
    assert result == "inserted"


@pytest.mark.asyncio
async def test_same_user_private_dedup_still_works(db, monkeypatch):
    # Seed an alice-private memory.
    existing = _mem("m-1", "deploy via argocd", visibility=PRIVATE, owner="u-alice")
    await db.insert_memory(existing)

    adapters = build_sqlite_adapters(db, memory_collection=_FakeColl(seeded=["m-1"]))
    store = MemoryStore(adapters.relational, adapters.keyword, adapters.vector, embed_cfg={}, dedup_threshold=0.08)

    async def _stub_embed(text):
        return [0.1, 0.1, 0.1]

    monkeypatch.setattr(store, "_embed", _stub_embed)

    duplicate = _mem("m-2", "deploy via argocd", visibility=PRIVATE, owner="u-alice")
    await _seed_doc(db, "d-3")
    result = await store.deduplicate_and_insert(
        duplicate,
        doc_id="d-3",
        source_type="agent_session",
        source_updated_at=None,
    )
    assert result in {"corroborated", "skipped"}


@pytest.mark.asyncio
async def test_workspace_write_does_not_corroborate_other_project(db, monkeypatch):
    # Workspace memories dedup only within the same project_key. The vector
    # channel does not pre-filter by project, so a PAY write receives RISK
    # candidates and the dedup guard must reject them.
    risk = _mem("m-risk", "deploy via argocd", visibility=WORKSPACE, project_key="RISK")
    await db.insert_memory(risk)

    adapters = build_sqlite_adapters(db, memory_collection=_FakeColl(seeded=["m-risk"]))
    store = MemoryStore(adapters.relational, adapters.keyword, adapters.vector, embed_cfg={}, dedup_threshold=0.08)

    async def _stub_embed(text):
        return [0.1, 0.1, 0.1]

    monkeypatch.setattr(store, "_embed", _stub_embed)

    pay = _mem("m-pay", "deploy via argocd", visibility=WORKSPACE, project_key="PAY")
    await _seed_doc(db, "d-pay")
    result = await store.deduplicate_and_insert(
        pay,
        doc_id="d-pay",
        source_type="confluence",
        source_updated_at=None,
    )
    assert result == "inserted"  # NOT "corroborated"


@pytest.mark.asyncio
async def test_private_write_does_not_corroborate_other_users_private(db, monkeypatch):
    # Two distinct private owners must not collide. Even though the writer
    # scope is keyed on memory.owner_user_id, the in-process candidate guard
    # is the second line of defense.
    bob = _mem("m-bob", "deploy via argocd", visibility=PRIVATE, owner="u-bob")
    await db.insert_memory(bob)

    adapters = build_sqlite_adapters(db, memory_collection=_FakeColl(seeded=["m-bob"]))
    store = MemoryStore(adapters.relational, adapters.keyword, adapters.vector, embed_cfg={}, dedup_threshold=0.08)

    async def _stub_embed(text):
        return [0.1, 0.1, 0.1]

    monkeypatch.setattr(store, "_embed", _stub_embed)

    alice = _mem("m-alice", "deploy via argocd", visibility=PRIVATE, owner="u-alice")
    await _seed_doc(db, "d-alice")
    result = await store.deduplicate_and_insert(
        alice,
        doc_id="d-alice",
        source_type="agent_session",
        source_updated_at=None,
    )
    assert result == "inserted"


@pytest.mark.asyncio
async def test_projected_equivalence_uses_exact_claim_not_vector_proximity(db, monkeypatch):
    incumbent = _mem("m-incumbent", "A7 is retained.", project_key="RISK")
    await db.insert_memory(incumbent)
    adapters = build_sqlite_adapters(
        db,
        memory_collection=_FakeColl(seeded=[incumbent.id]),
    )
    store = MemoryStore(
        adapters.relational,
        adapters.keyword,
        adapters.vector,
        embed_cfg={},
        dedup_threshold=0.08,
    )

    async def _stub_embed(_text):
        return [0.1, 0.1, 0.1]

    monkeypatch.setattr(store, "_embed", _stub_embed)

    conflicting = _mem("m-conflict", "A7 is removed.", project_key="PAY")

    assert [item.id for item in await store.find_access_compatible_equivalence_candidates(conflicting)] == [
        incumbent.id
    ]


@pytest.mark.asyncio
async def test_projected_equivalence_crosses_project_relevance_boundary(db, monkeypatch):
    incumbent = _mem("m-incumbent", "A7 is retained.", project_key="RISK")
    await db.insert_memory(incumbent)
    adapters = build_sqlite_adapters(
        db,
        memory_collection=_FakeColl(seeded=[incumbent.id]),
    )
    store = MemoryStore(
        adapters.relational,
        adapters.keyword,
        adapters.vector,
        embed_cfg={},
        dedup_threshold=0.08,
    )

    async def _stub_embed(_text):
        return [0.1, 0.1, 0.1]

    monkeypatch.setattr(store, "_embed", _stub_embed)

    same_claim = _mem("m-equivalent", "A7 is retained.", project_key="PAY")
    candidates = await store.find_access_compatible_equivalence_candidates(same_claim)

    assert [item.id for item in candidates] == [incumbent.id]


@pytest.mark.asyncio
async def test_projected_equivalence_recalls_exact_rebaseline_retirement_without_vector(
    db,
    monkeypatch,
):
    retired = _mem(
        "m-rebaseline-retired",
        "A7 is retained.",
        project_key="RISK",
        status="retired",
    )
    retired.retirement_reason = "source_rebaseline"
    await db.insert_memory(retired)
    adapters = build_sqlite_adapters(db, memory_collection=_FakeColl())
    store = MemoryStore(
        adapters.relational,
        adapters.keyword,
        adapters.vector,
        embed_cfg={},
        dedup_threshold=0.08,
    )

    async def _stub_embed(_text):
        return [0.1, 0.1, 0.1]

    monkeypatch.setattr(store, "_embed", _stub_embed)
    replayed = _mem("m-replayed", "A7 is retained.", project_key="PAY")

    candidates = await store.find_access_compatible_equivalence_candidates(replayed)

    assert [item.id for item in candidates] == [retired.id]


@pytest.mark.asyncio
async def test_projected_equivalence_does_not_cross_repository_access_identity(db, monkeypatch):
    incumbent = _mem("m-incumbent", "A7 is retained.", project_key="RISK")
    incumbent.repo_identifier = "repo-a"
    await db.insert_memory(incumbent)
    adapters = build_sqlite_adapters(
        db,
        memory_collection=_FakeColl(seeded=[incumbent.id]),
    )
    store = MemoryStore(
        adapters.relational,
        adapters.keyword,
        adapters.vector,
        embed_cfg={},
        dedup_threshold=0.08,
    )

    async def _stub_embed(_text):
        return [0.1, 0.1, 0.1]

    monkeypatch.setattr(store, "_embed", _stub_embed)
    candidate = _mem("m-equivalent", "A7 is retained.", project_key="PAY")
    candidate.repo_identifier = "repo-b"

    assert await store.find_access_compatible_equivalence_candidates(candidate) == ()
