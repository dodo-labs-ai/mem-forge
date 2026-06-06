"""Run the parameterizable contract suite against the in-repo SQLite/FTS/Chroma
adapters.

This is the single in-repo consumer of ``tests/contracts``: every behavior
asserted in the abstract base classes runs here against the same
``SqliteRelationalStore``, ``SqliteKeywordSearch``, and ``SqliteVectorStore``
the live API binds. A future non-SQLite adapter family ships its own
mirror of this module without repeating any of the test bodies themselves.
"""

from __future__ import annotations

import uuid

import chromadb
import pytest

from memforge.storage.adapters.sqlite.factory import build_sqlite_adapters
from memforge.storage.database import Database

from tests.contracts._support import (
    AdaptersFactory,
    ContractAdapters,
    FactoryResult,
)
from tests.contracts.test_keyword_contract import KeywordSearchContract
from tests.contracts.test_relational_contract import RelationalStoreContract
from tests.contracts.test_vector_contract import VectorStoreContract


def _sqlite_adapters_factory(tmp_path) -> AdaptersFactory:
    """Build a per-test factory that hands the suite a fresh SQLite+Chroma bundle.

    Each call connects a brand-new SQLite database under the test's
    ``tmp_path`` and creates a brand-new Chroma collection on a single
    ephemeral Chroma client per test. The teardown closes the database
    and drops the collection so no row from one test reaches another.
    """
    chroma_client = chromadb.EphemeralClient()

    async def factory() -> FactoryResult:
        # A unique collection name per call keeps multiple invocations of
        # the factory inside one test (rare, but the contract API does not
        # forbid it) on disjoint Chroma state.
        collection_name = f"contract-{uuid.uuid4().hex[:12]}"
        database = Database(str(tmp_path / f"contract-{uuid.uuid4().hex[:8]}.db"))
        await database.connect()
        collection = chroma_client.get_or_create_collection(collection_name)
        adapters = build_sqlite_adapters(database, collection)
        bundle = ContractAdapters(
            relational=adapters.relational,
            keyword=adapters.keyword,
            vector=adapters.vector,
        )

        async def teardown() -> None:
            await database.close()
            try:
                chroma_client.delete_collection(collection_name)
            except Exception:
                # The contract suite never asserts on Chroma teardown
                # behavior; an EphemeralClient drops its state when the
                # process exits anyway.
                pass

        return FactoryResult(adapters=bundle, teardown=teardown)

    return factory


class TestSqliteRelationalStoreContract(RelationalStoreContract):
    @pytest.fixture
    def adapters_factory(self, tmp_path) -> AdaptersFactory:
        return _sqlite_adapters_factory(tmp_path)


class TestSqliteKeywordSearchContract(KeywordSearchContract):
    @pytest.fixture
    def adapters_factory(self, tmp_path) -> AdaptersFactory:
        return _sqlite_adapters_factory(tmp_path)


class TestSqliteVectorStoreContract(VectorStoreContract):
    @pytest.fixture
    def adapters_factory(self, tmp_path) -> AdaptersFactory:
        return _sqlite_adapters_factory(tmp_path)
