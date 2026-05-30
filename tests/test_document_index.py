from __future__ import annotations

import numpy as np

from meminception.retrieval.document_index import DocumentVectorIndex


class NumpyEmbeddingCollection:
    def get(self, *, ids=None, include=None):
        return {
            "ids": ["doc-1"],
            "embeddings": np.array([[0.1, 0.2, 0.3]]),
            "documents": ["semantic text"],
            "metadatas": [{"content_hash": "hash-1", "version": "1"}],
        }


def test_snapshot_accepts_numpy_embedding_results():
    snapshot = DocumentVectorIndex(NumpyEmbeddingCollection()).snapshot("doc-1")

    assert snapshot["id"] == "doc-1"
    assert snapshot["embedding"].tolist() == [0.1, 0.2, 0.3]
    assert snapshot["document"] == "semantic text"
    assert snapshot["metadata"] == {"content_hash": "hash-1", "version": "1"}
