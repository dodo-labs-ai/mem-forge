"""Stand-ins for the Memory vector index the default runtime opens."""

from __future__ import annotations

from typing import Any

from memforge.retrieval import vector_index
from memforge.retrieval.vector_index import MemoryVectorIndex


def use_collection_as_index(monkeypatch, collection: Any, *, fingerprint: str | None = None) -> list[str]:
    """Make ``collection`` the Memory vector index in use for every database.

    Whoever opens the index gets this collection, bound to ``fingerprint``;
    without one it is bound to no embedding model and takes any vector.
    Returns the list the Chroma path of every opening is recorded in.
    """
    opened: list[str] = []

    async def open_index(_db: Any, chroma_path: str) -> MemoryVectorIndex:
        opened.append(chroma_path)
        return MemoryVectorIndex(collection=collection, fingerprint=fingerprint)

    monkeypatch.setattr(vector_index, "open_memory_vector_index", open_index)
    return opened


def use_no_index(monkeypatch) -> None:
    """Leave every database without a Memory vector index in use, whatever it records."""

    async def open_index(_db: Any, _chroma_path: str) -> None:
        return None

    monkeypatch.setattr(vector_index, "open_memory_vector_index", open_index)


class EmbeddingModels:
    """Stand in for the embedding provider behind LiteLLM.

    Each model answers with vectors of its own length, and no two models
    answer one text with the same vector. Every request is recorded. A failure
    that is set is raised for each request until it is cleared, and
    ``before_answer`` runs before a request is answered.
    """

    def __init__(self, monkeypatch, **dimensions: int) -> None:
        self.dimensions = dict(dimensions)
        self.requests: list[dict[str, Any]] = []
        self.failure: Exception | None = None
        self.before_answer: Any = None
        monkeypatch.setattr("litellm.aembedding", self)

    @property
    def texts(self) -> list[str]:
        return [text for request in self.requests for text in request["input"]]

    def models(self) -> list[str]:
        return [request["model"] for request in self.requests]

    def vector(self, model: str, text: str) -> list[float]:
        name = model.rpartition("/")[2]
        dimension = self.dimensions[name]
        marker = float(sum(map(ord, name)) % 11 + 1)
        return [float(len(text) % 7 + 1), marker, *([0.5] * (dimension - 2))]

    async def __call__(self, **request: Any) -> dict[str, Any]:
        self.requests.append(request)
        if self.before_answer is not None:
            await self.before_answer(request)
        if self.failure is not None:
            raise self.failure
        return {
            "model": request["model"].rpartition("/")[2],
            "data": [
                {"index": index, "embedding": self.vector(request["model"], text)}
                for index, text in enumerate(request["input"])
            ],
        }
