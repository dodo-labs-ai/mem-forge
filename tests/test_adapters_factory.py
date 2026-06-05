"""build_sqlite_adapters wires a Database plus collection into the three impls."""

from __future__ import annotations

import pytest

from memforge.storage.database import Database
from memforge.storage.adapters.protocols import (
    KeywordSearch,
    RelationalStore,
    VectorStore,
)
from memforge.storage.adapters.sqlite.factory import build_sqlite_adapters


class FakeCollection:
    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

    def upsert(self, **kwargs):
        pass

    def delete(self, **kwargs):
        pass

    def get(self, **kwargs):
        return {"ids": []}


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "factory.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_build_sqlite_adapters_returns_protocol_conforming_impls(db):
    adapters = build_sqlite_adapters(db, FakeCollection())
    assert isinstance(adapters.relational, RelationalStore)
    assert isinstance(adapters.keyword, KeywordSearch)
    assert isinstance(adapters.vector, VectorStore)


@pytest.mark.asyncio
async def test_vector_handle_wraps_the_passed_collection(db):
    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    assert adapters.vector.collection is collection
