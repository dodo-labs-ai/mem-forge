"""Keyword search contract.

The base class is parameterized by an ``AdaptersFactory`` and reads against
``ContractAdapters.keyword`` plus ``ContractAdapters.relational`` (only for
inserting the rows the keyword channel matches against). The relational
write path is shared with the SQLite/FTS5 case where memory inserts also
update the FTS5 index in the same transaction; the contract here only
asserts on what the keyword channel observes after a normal write.
"""

from __future__ import annotations

import pytest

from memforge.retrieval.filters import MemorySourceFilter
from memforge.storage.adapters.protocols import KeywordSearch

from tests.contracts._support import (
    AdaptersFactory,
    ContractAdapters,
    FactoryResult,
    make_document,
    make_memory,
    make_scope,
)


class KeywordSearchContract:
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

    async def test_satisfies_keyword_search_protocol(self, adapters: ContractAdapters) -> None:
        assert isinstance(adapters.keyword, KeywordSearch)

    async def test_search_matches_active_memory_by_content(self, adapters: ContractAdapters) -> None:
        await adapters.relational.insert_memory(make_memory("m1", content="PostgreSQL connection pooling"))
        hits = await adapters.keyword.search('"PostgreSQL"', make_scope(), None, limit=10)
        assert [mid for mid, _ in hits] == ["m1"]

    async def test_search_filters_by_status_via_scope(self, adapters: ContractAdapters) -> None:
        await adapters.relational.insert_memory(
            make_memory("m-active", content="Redis cache eviction", status="active")
        )
        await adapters.relational.insert_memory(
            make_memory("m-retired", content="Redis cache eviction", status="retired")
        )
        hits = await adapters.keyword.search('"Redis"', make_scope(), None, limit=10)
        assert [mid for mid, _ in hits] == ["m-active"]

    async def test_search_filters_by_memory_type(self, adapters: ContractAdapters) -> None:
        await adapters.relational.insert_memory(make_memory("m1", content="deploy via ArgoCD", memory_type="fact"))
        empty = await adapters.keyword.search('"deploy"', make_scope(), ["decision"], limit=10)
        assert empty == []
        kept = await adapters.keyword.search('"deploy"', make_scope(), ["fact"], limit=10)
        assert [mid for mid, _ in kept] == ["m1"]

    async def test_remove_drops_the_row_from_keyword_results(self, adapters: ContractAdapters) -> None:
        await adapters.relational.insert_memory(make_memory("m1", content="PostgreSQL connection pooling"))
        before = await adapters.keyword.search('"PostgreSQL"', make_scope(), None, limit=10)
        assert before != []
        await adapters.keyword.remove("m1")
        after = await adapters.keyword.search('"PostgreSQL"', make_scope(), None, limit=10)
        assert after == []

    async def test_metadata_title_tokens_recall_memory(self, adapters: ContractAdapters) -> None:
        await adapters.relational.insert_memory(
            make_memory(
                "m-blocker",
                content="Lifecycle assignment skips person assignment creation",
            )
        )
        await adapters.relational.upsert_document(
            make_document(
                "SFPAY-179397",
                source="src-jira",
                title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
                labels=["lifecycle"],
            )
        )
        await adapters.relational.add_memory_source(
            "m-blocker",
            "SFPAY-179397",
            "jira",
            None,
            support_kind="extracted",
            source_updated_at=None,
        )

        hits = await adapters.keyword.search_metadata('"create" "blocker" "hint"', make_scope(), None, limit=10)

        assert [hit.memory_id for hit in hits] == ["m-blocker"]
        assert hits[0].channel == "bm25_metadata_tokens"
        assert "metadata_any" in hits[0].matched_fields
        assert hits[0].source_refs[0].doc_id == "SFPAY-179397"

    async def test_metadata_source_filter_applies_to_matching_support_row(self, adapters: ContractAdapters) -> None:
        await adapters.relational.insert_memory(
            make_memory(
                "m-shared",
                content="Lifecycle assignment skips person assignment creation",
            )
        )
        await adapters.relational.upsert_document(
            make_document(
                "SFPAY-179397",
                source="src-jira",
                title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
            )
        )
        await adapters.relational.upsert_document(
            make_document(
                "wiki-runbook",
                source="src-wiki",
                title="Payroll lifecycle runbook",
            )
        )
        await adapters.relational.add_memory_source(
            "m-shared",
            "SFPAY-179397",
            "jira",
            None,
            support_kind="extracted",
            source_updated_at=None,
        )
        await adapters.relational.add_memory_source(
            "m-shared",
            "wiki-runbook",
            "confluence",
            None,
            support_kind="extracted",
            source_updated_at=None,
        )

        hits = await adapters.keyword.search_metadata(
            '"create" "blocker" "hint"',
            make_scope(),
            None,
            limit=10,
            source_filter=MemorySourceFilter(source_ids=("src-wiki",)),
        )

        assert hits == []

    async def test_metadata_client_filter_matches_document_client(self, adapters: ContractAdapters) -> None:
        await adapters.relational.insert_memory(
            make_memory(
                "m-codex",
                content="Agent session summary",
            )
        )
        await adapters.relational.upsert_document(
            make_document(
                "codex-session-1",
                source="src-codex",
                client="codex",
                title="Create Blocker Hint investigation",
            )
        )
        await adapters.relational.add_memory_source(
            "m-codex",
            "codex-session-1",
            "agent_session",
            None,
            support_kind="extracted",
            source_updated_at=None,
        )

        hits = await adapters.keyword.search_metadata(
            '"create" "blocker" "hint"',
            make_scope(),
            None,
            limit=10,
            source_filter=MemorySourceFilter(clients=("codex",)),
        )

        assert [hit.memory_id for hit in hits] == ["m-codex"]
        assert (
            await adapters.keyword.search_metadata(
                '"create" "blocker" "hint"',
                make_scope(),
                None,
                limit=10,
                source_filter=MemorySourceFilter(clients=("claude-code",)),
            )
            == []
        )
