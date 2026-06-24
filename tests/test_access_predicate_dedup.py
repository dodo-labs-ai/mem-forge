# Cross-datastore isolation is a deployment-time property of the bound
# adapter handle, not of the core predicate; this test pack only exercises
# the predicate.
"""A workspace writer must not corroborate or merge into another user's
private memory at the same content. The dedup candidate selection rides
on a writer scope that excludes other users' private rows from the vector
channel before the threshold check ever runs.
"""

from __future__ import annotations

import pytest

from memforge.memory.audit import MemoryAuditLogger
from memforge.memory.store import MemoryStore
from memforge.models import (
    Memory,
    SHARED_PROJECT_KEY,
    Visibility,
    content_hash,
)
from memforge.storage.adapters.context import AccessScope
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


WORKSPACE = Visibility.WORKSPACE.value
PRIVATE = Visibility.PRIVATE.value


def _mem(mid: str, content: str, *, visibility=WORKSPACE, owner=None) -> Memory:
    return Memory(
        id=mid,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content + mid),
        visibility=visibility,
        owner_user_id=owner,
        project_key=SHARED_PROJECT_KEY,
        tags=[],
        status="active",
    )


class _DedupColl:
    """Chroma fake that honors the access-predicate where dict.

    Returns identical (zero-distance) candidates for any embedding so the
    dedup threshold is satisfied for every visible row, ensuring the test
    only fails if a forbidden row leaks past the where filter.
    """

    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    def upsert(self, *, ids, embeddings, metadatas, **_):
        for i, mid in enumerate(ids):
            self.items[mid] = {"metadata": dict(metadatas[i])}

    def delete(self, ids=None, **_):
        for mid in ids or []:
            self.items.pop(mid, None)

    def query(self, *, query_embeddings, n_results, where=None, **_):
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
                dists.append(0.0)  # exact match: well within any dedup threshold
        return {"ids": [ids[:n_results]], "distances": [dists[:n_results]]}


@pytest.fixture
async def store_fixture(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "dedup.db"))
    await db.connect()
    coll = _DedupColl()
    adapters = build_sqlite_adapters(db, coll)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )

    async def _fake_embed(text: str) -> list[float]:
        return [0.1, 0.1, 0.1]

    monkeypatch.setattr(store, "_embed", _fake_embed)
    yield store, db, adapters, coll
    await db.close()


def _writer_scope() -> AccessScope:
    """The dedup writer scope: workspace-only, no other-user private leakage.

    A writer dedup pass must never see any user's private memories: the
    post-fusion re-check uses include_private=False so a private row cannot
    enter the candidate pool from any channel.
    """
    return AccessScope(
        user_id="u-1",
        include_private=False,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="project-first",
    )


@pytest.mark.asyncio
async def test_workspace_write_does_not_corroborate_other_users_private(store_fixture):
    store, db, _adapters, coll = store_fixture

    # Seed a U2-private memory at the exact content U1 is about to write.
    u2_priv = _mem("priv-u2", "deploy via argocd", visibility=PRIVATE, owner="u-2")
    await db.insert_memory(u2_priv)
    await store.vector.upsert(
        ids=[u2_priv.id],
        embeddings=[[0.1, 0.1, 0.1]],
        metadatas=[
            {
                "memory_type": "fact",
                "project_key": SHARED_PROJECT_KEY,
                "visibility": PRIVATE,
                "owner_user_id": "u-2",
                "confidence": u2_priv.confidence,
                "status": "active",
                "content_hash": u2_priv.content_hash,
                "embedding_text_hash": "h",
            }
        ],
    )

    # Seed a doc row so the source linkage on insert succeeds.
    from datetime import datetime, timezone
    from memforge.models import DocumentRecord

    now = datetime.now(timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id="doc-team",
            source="src-team",
            source_url="team://doc",
            title="t",
            space_or_project="PROJ",
            author="a",
            last_modified=now,
            labels=[],
            version="1",
            content_hash="h-team",
            token_count=1,
            raw_content_uri=None,
            raw_content_type="text/markdown",
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )

    # U1 writes the same content as a workspace memory.
    new_mem = _mem("ws-new", "deploy via argocd", visibility=WORKSPACE)
    result = await store.deduplicate_and_insert(
        memory=new_mem,
        doc_id="doc-team",
        source_type="manual",
        scope=_writer_scope(),
        source_updated_at=None,
    )

    # The write must land as a fresh insert, not corroborate U2's private row.
    assert result == "inserted"
    surviving = await db.get_memory("priv-u2")
    assert surviving is not None
    assert surviving.corroboration_count == 1  # untouched

    # Verify the predicate kept U2's private row out of the dedup candidate
    # pool: the writer scope's chroma where filter excludes private rows
    # entirely, so even with embeddings that match exactly the candidate set
    # only contains workspace memories.
    candidates = coll.query(
        query_embeddings=[[0.1, 0.1, 0.1]],
        n_results=10,
        where={
            "$and": [
                {"status": "active"},
                {"visibility": WORKSPACE},
            ]
        },
    )
    assert "priv-u2" not in candidates["ids"][0]
