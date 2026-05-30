from __future__ import annotations

from typing import Any

from memforge.memory.index_payloads import embedding_vector_hash
from memforge.retrieval.vector_metadata import upsert_with_stored_vector_hash


class RoundTripCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.updated: list[str] = []
        self.upserts: list[dict[str, Any]] = []

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        for index, record_id in enumerate(ids):
            self.upserts.append(dict(metadatas[index] if metadatas else {}))
            stored_embedding = None
            if embeddings is not None:
                stored_embedding = [float(value) + 0.000000001 for value in embeddings[index]]
            self.records[record_id] = {
                "embedding": stored_embedding,
                "metadata": dict(metadatas[index] if metadatas else {}),
                "document": documents[index] if documents else None,
            }

    def get(self, *, ids=None, include=None):
        selected = [record_id for record_id in (ids or self.records) if record_id in self.records]
        include = include or []
        result: dict[str, Any] = {"ids": selected}
        if "embeddings" in include:
            result["embeddings"] = [self.records[record_id]["embedding"] for record_id in selected]
        if "metadatas" in include:
            result["metadatas"] = [self.records[record_id]["metadata"] for record_id in selected]
        if "documents" in include:
            result["documents"] = [self.records[record_id]["document"] for record_id in selected]
        return result

    def update(self, *, ids, metadatas) -> None:
        for index, record_id in enumerate(ids):
            self.records[record_id]["metadata"] = dict(metadatas[index])
            self.updated.append(record_id)


def test_upsert_with_stored_vector_hash_uses_chroma_round_tripped_embedding():
    collection = RoundTripCollection()
    attempted_embedding = [0.1, 0.2, 0.3]

    upsert_with_stored_vector_hash(
        collection,
        ids=["item-1"],
        embeddings=[attempted_embedding],
        metadatas=[{
            "embedding_vector_hash": embedding_vector_hash(attempted_embedding),
            "content_hash": "content",
        }],
        documents=["semantic text"],
    )

    stored_embedding = collection.records["item-1"]["embedding"]
    assert collection.updated == ["item-1"]
    assert collection.records["item-1"]["metadata"]["embedding_vector_hash"] == (
        embedding_vector_hash(stored_embedding)
    )
    assert collection.records["item-1"]["metadata"]["embedding_vector_hash"] != (
        embedding_vector_hash(attempted_embedding)
    )


def test_upsert_with_stored_vector_hash_strips_attempted_vector_hash_before_initial_write():
    collection = RoundTripCollection()
    attempted_embedding = [0.1, 0.2, 0.3]

    upsert_with_stored_vector_hash(
        collection,
        ids=["item-1"],
        embeddings=[attempted_embedding],
        metadatas=[{
            "embedding_vector_hash": "caller-owned-wrong-hash",
            "content_hash": "content",
        }],
    )

    assert "embedding_vector_hash" not in collection.upserts[0]
    assert collection.records["item-1"]["metadata"]["embedding_vector_hash"] != "caller-owned-wrong-hash"
