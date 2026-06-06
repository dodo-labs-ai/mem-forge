"""SqliteVectorStore: the Chroma wrapper that owns distance-to-score."""

from __future__ import annotations

from typing import Any

import pytest

from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID
from memforge.storage.adapters.protocols import VectorStore
from memforge.storage.adapters.sqlite.vector import SqliteVectorStore


def _scope() -> AccessScope:
    return AccessScope(
        user_id=LOCAL_DEV_USER_ID,
        include_private=False,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="project-first",
    )


class FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.last_query: dict[str, Any] | None = None
        self.deleted: list[str] = []

    def query(self, **kwargs):
        self.last_query = kwargs
        ids = list(self.records)
        return {"ids": [ids], "distances": [[0.25 for _ in ids]]}

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None):
        for index, record_id in enumerate(ids):
            self.records[record_id] = {
                "embedding": embeddings[index] if embeddings else None,
                "metadata": dict(metadatas[index]) if metadatas else {},
            }

    def delete(self, *, ids):
        self.deleted.extend(ids)
        for record_id in ids:
            self.records.pop(record_id, None)

    def get(self, *, ids=None, include=None):
        selected = [r for r in (ids or list(self.records)) if r in self.records]
        out: dict[str, Any] = {"ids": selected}
        include = include or []
        if "metadatas" in include:
            out["metadatas"] = [self.records[r]["metadata"] for r in selected]
        if "embeddings" in include:
            out["embeddings"] = [self.records[r]["embedding"] for r in selected]
        return out


def test_satisfies_vector_store_protocol():
    store = SqliteVectorStore(FakeCollection())
    assert isinstance(store, VectorStore)


def test_declares_cosine_metric():
    store = SqliteVectorStore(FakeCollection())
    assert store.distance_metric == "cosine"


def test_similarity_is_one_minus_distance():
    store = SqliteVectorStore(FakeCollection())
    assert store.similarity(0.0) == pytest.approx(1.0)
    assert store.similarity(0.25) == pytest.approx(0.75)


def test_similarity_is_floored_at_zero_for_distances_past_one():
    store = SqliteVectorStore(FakeCollection())
    assert store.similarity(1.5) == 0.0


def test_similarity_is_monotonic_decreasing_in_distance():
    store = SqliteVectorStore(FakeCollection())
    assert store.similarity(0.1) > store.similarity(0.4) > store.similarity(0.9)


def test_within_dedup_threshold_owns_the_distance_comparison():
    store = SqliteVectorStore(FakeCollection())
    # A distance threshold of 0.08 means scores at distance < 0.08 are dupes.
    # score 0.95 -> distance 0.05 (< 0.08) -> within threshold.
    assert store.within_dedup_threshold(0.08, 0.95) is True
    # score 0.90 -> distance 0.10 (>= 0.08) -> not within threshold.
    assert store.within_dedup_threshold(0.08, 0.90) is False


@pytest.mark.asyncio
async def test_query_returns_id_score_pairs_via_similarity():
    collection = FakeCollection()
    store = SqliteVectorStore(collection)
    await store.upsert(["m1", "m2"], [[0.1], [0.2]], [{}, {}])
    pairs = await store.query([0.1], _scope(), None, limit=10)
    assert {mid for mid, _ in pairs} == {"m1", "m2"}
    # distance 0.25 -> similarity 0.75 (the conversion lives here, not in the caller)
    assert all(score == pytest.approx(0.75) for _, score in pairs)


@pytest.mark.asyncio
async def test_query_propagates_backend_failure_for_fail_closed_dedup():
    class FailingCollection(FakeCollection):
        def query(self, **kwargs):
            raise RuntimeError("chroma query failed")

    store = SqliteVectorStore(FailingCollection())
    # query never swallows: the dedup write path relies on this raise to
    # abort and audit (it must not silently treat a backend failure as
    # "no duplicates found").
    with pytest.raises(RuntimeError, match="chroma query failed"):
        await store.query([0.1], _scope(), None, limit=10)


@pytest.mark.asyncio
async def test_delete_removes_records():
    collection = FakeCollection()
    store = SqliteVectorStore(collection)
    await store.upsert(["m1"], [[0.1]], [{}])
    await store.delete(["m1"])
    assert collection.deleted == ["m1"]
    assert await store.get_record("m1") is None


@pytest.mark.asyncio
async def test_get_record_returns_embedding_and_metadata():
    collection = FakeCollection()
    store = SqliteVectorStore(collection)
    await store.upsert(["m1"], [[0.1, 0.2]], [{"memory_type": "fact"}])
    record = await store.get_record("m1")
    assert record is not None
    stored_metadata = dict(record["metadata"])
    stored_metadata.pop("embedding_vector_hash", None)  # stamped by upsert_with_stored_vector_hash, not part of the caller payload
    assert stored_metadata == {"memory_type": "fact"}
