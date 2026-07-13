"""Vector store contract.

The base class is parameterized by an ``AdaptersFactory`` and reads against
``ContractAdapters.vector`` only. Distance-to-similarity, dedup-threshold,
and round-trip behavior are uniform across adapters; the suite never
asserts on absolute ranking quality, only on the conversions the protocol
itself owns and on the visibility/scope filtering each backend must apply.
"""

from __future__ import annotations

import pytest

from memforge.models import Visibility
from memforge.storage.adapters.protocols import VectorStore

from tests.contracts._support import (
    AdaptersFactory,
    ContractAdapters,
    DEFAULT_EMBEDDING,
    FactoryResult,
    make_memory,
    make_scope,
    make_vector_metadata,
)


class VectorStoreContract:
    """Subclass and override ``adapters_factory`` to bind a concrete adapter."""

    @pytest.fixture
    def adapters_factory(self) -> AdaptersFactory:  # pragma: no cover - subclass override
        raise NotImplementedError(
            "Subclass must override adapters_factory with an async callable that returns FactoryResult"
        )

    @pytest.fixture
    async def adapters(self, adapters_factory: AdaptersFactory) -> ContractAdapters:
        result: FactoryResult = await adapters_factory()
        try:
            yield result.adapters
        finally:
            await result.teardown()

    # -- Protocol conformance -----------------------------------------------

    def test_satisfies_vector_store_protocol(self, adapters: ContractAdapters) -> None:
        assert isinstance(adapters.vector, VectorStore)

    def test_declares_a_distance_metric(self, adapters: ContractAdapters) -> None:
        assert isinstance(adapters.vector.distance_metric, str)
        assert adapters.vector.distance_metric

    # -- Distance-to-score conversions --------------------------------------

    def test_similarity_is_floored_at_zero(self, adapters: ContractAdapters) -> None:
        # A distance past the metric's max range must not yield a negative
        # similarity; rank fusion treats negatives as meaningless.
        assert adapters.vector.similarity(99.0) >= 0.0

    def test_similarity_is_monotonic_decreasing_in_distance(self, adapters: ContractAdapters) -> None:
        v = adapters.vector
        a = v.similarity(0.1)
        b = v.similarity(0.4)
        c = v.similarity(0.9)
        assert a >= b >= c

    def test_within_dedup_threshold_owns_the_distance_comparison(self, adapters: ContractAdapters) -> None:
        v = adapters.vector
        # A score that is above (1 - threshold) is within the dedup
        # threshold; one that is below is not. Exact distances are
        # backend-specific, but this band-around-the-edge property holds
        # for any monotonic distance-to-score.
        threshold = 0.08
        # similarity(0) is the maximum-possible score; that is always
        # within any positive dedup threshold.
        assert v.within_dedup_threshold(threshold, v.similarity(0.0)) is True
        # A score that maps from a distance well past the threshold must
        # not be flagged as a duplicate.
        assert v.within_dedup_threshold(threshold, v.similarity(0.5)) is False

    # -- Round-trip ---------------------------------------------------------

    async def test_upsert_then_get_record_round_trips_metadata(self, adapters: ContractAdapters) -> None:
        memory = make_memory("m1")
        await adapters.vector.upsert(["m1"], [list(DEFAULT_EMBEDDING)], [make_vector_metadata(memory)])
        record = await adapters.vector.get_record("m1")
        assert record is not None
        metadata = dict(record["metadata"])
        # Adapters may stamp implementation-private keys (e.g. an embedding
        # hash); the contract only asserts on the keys the caller wrote.
        for key, expected in make_vector_metadata(memory).items():
            assert metadata.get(key) == expected

    async def test_get_record_returns_none_for_missing_id(self, adapters: ContractAdapters) -> None:
        assert await adapters.vector.get_record("does-not-exist") is None

    async def test_delete_removes_the_record(self, adapters: ContractAdapters) -> None:
        memory = make_memory("m1")
        await adapters.vector.upsert(["m1"], [list(DEFAULT_EMBEDDING)], [make_vector_metadata(memory)])
        await adapters.vector.delete(["m1"])
        assert await adapters.vector.get_record("m1") is None

    # -- Visibility filtering on query -------------------------------------

    async def test_query_hides_other_users_private_rows(self, adapters: ContractAdapters) -> None:
        owner_private = make_memory(
            "owner-private",
            visibility=Visibility.PRIVATE.value,
            owner_user_id="dev",
        )
        other_private = make_memory(
            "other-private",
            visibility=Visibility.PRIVATE.value,
            owner_user_id="someone-else",
        )
        await adapters.vector.upsert(
            ["owner-private", "other-private"],
            [list(DEFAULT_EMBEDDING), list(DEFAULT_EMBEDDING)],
            [
                make_vector_metadata(owner_private),
                make_vector_metadata(other_private),
            ],
        )
        scope = make_scope(include_private=True)
        hits = await adapters.vector.query(list(DEFAULT_EMBEDDING), scope, None, limit=10)
        assert {mid for mid, _ in hits} == {"owner-private"}

    async def test_query_drops_private_rows_when_scope_excludes_them(self, adapters: ContractAdapters) -> None:
        workspace_row = make_memory("workspace1")
        private_row = make_memory(
            "private1",
            visibility=Visibility.PRIVATE.value,
            owner_user_id="dev",
        )
        await adapters.vector.upsert(
            ["workspace1", "private1"],
            [list(DEFAULT_EMBEDDING), list(DEFAULT_EMBEDDING)],
            [
                make_vector_metadata(workspace_row),
                make_vector_metadata(private_row),
            ],
        )
        hits = await adapters.vector.query(list(DEFAULT_EMBEDDING), make_scope(), None, limit=10)
        assert {mid for mid, _ in hits} == {"workspace1"}
