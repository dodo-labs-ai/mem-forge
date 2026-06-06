# Cross-datastore isolation is a deployment-time property of the bound
# adapter handle, not of the core predicate; this test pack only exercises
# the predicate.
"""The service-path search engine has no document fallback.

`build_search_engine` constructs the engine without a `document_vector`,
so `_document_fallback` returns an empty list regardless of the query or
the remaining slot count. The fallback exists for offline tooling that
constructs the engine directly, not for the live service surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memforge import runtime
from memforge.config import AppConfig
from memforge.storage.database import Database


class FakeCollection:
    def query(self, **kwargs):
        return {"ids": [["should-not-surface"]], "distances": [[0.0]]}

    def upsert(self, **kwargs):
        return None

    def delete(self, **kwargs):
        return None


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "doc_fallback.db"))
    await database.connect()
    yield database
    await database.close()


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.llm.embedding_base_url = "http://localhost:6655/openai/v1"
    cfg.llm.embedding_api_key = "test-key"
    cfg.server.jwt_secret = "test-secret"
    return cfg


@pytest.mark.asyncio
async def test_build_search_engine_does_not_wire_document_collection(
    db, tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        runtime, "get_chroma_collection", lambda **kwargs: FakeCollection(),
    )

    cfg = _config(tmp_path)
    engine = await runtime.build_search_engine(db, cfg)

    # The service path constructs the engine without a document vector.
    # `_document_fallback` returns [] in that mode, so a leaky fake document
    # collection cannot ride the fallback into the result list.
    assert engine._document_vector is None


@pytest.mark.asyncio
async def test_document_fallback_returns_empty_when_unwired(db, tmp_path, monkeypatch):
    monkeypatch.setattr(
        runtime, "get_chroma_collection", lambda **kwargs: FakeCollection(),
    )
    cfg = _config(tmp_path)
    engine = await runtime.build_search_engine(db, cfg)

    # Force an embedding so the early-return is the doc-vector branch, not
    # the unembeddable-query branch.
    engine._get_or_compute_embedding = lambda query: [0.1, 0.1, 0.1]

    fallback = await engine._document_fallback(
        query="anything", remaining_slots=5, exclude_doc_ids=set(),
    )
    assert fallback == []


def test_runtime_build_search_engine_passes_no_document_collection_kwarg():
    """A static check that the service builder does not pass `document_vector`
    to `SearchEngine`. A future refactor that wires it without a predicate
    pre-filter would silently re-enable a leak vector; this test pins the
    construction shape."""
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    # Locate the build_search_engine function body and verify it does not
    # mention document_vector or document_collection in its construction.
    marker = "async def build_search_engine"
    start = source.index(marker)
    end_marker = "\nasync def "
    end = source.find(end_marker, start + len(marker))
    body = source[start:end if end != -1 else len(source)]
    assert "document_vector" not in body
    assert "document_collection" not in body
