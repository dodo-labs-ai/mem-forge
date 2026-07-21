"""Relational store contract.

The base class is parameterized by an ``AdaptersFactory`` the consumer
provides via the ``adapters`` fixture. Every test in this module reads
against ``ContractAdapters.relational`` only; keyword/vector branches live
in their own contract modules so a relational-only adapter can consume
this one in isolation.

Invariants covered:

* round-trip (insert, get, list-by-source) for the canonical row write/read
* visibility filtering on the workspace branch and the private-owner branch
* lifecycle status filtering via ``AccessScope.allowed_statuses``
* scope-mode hard narrowing (``project`` vs ``project-first``)
* project lifecycle (create/list/update + project deletion that rebuckets
  named memories into UNSORTED in one transaction)

Invariants intentionally NOT covered here because the public protocol does
not expose them in a way every adapter can satisfy uniformly: graph 1-hop
expansion and the FTS-status-filter rebuild path. Both are exercised by
the existing SQLite-targeted tests, where the additional database hooks
are part of the concrete implementation.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.models import (
    SHARED_PROJECT_KEY,
    UNSORTED_PROJECT_KEY,
    Visibility,
)
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
from memforge.storage.adapters.protocols import RelationalStore

from tests.contracts._support import (
    AdaptersFactory,
    ContractAdapters,
    FactoryResult,
    make_document,
    make_memory,
    make_scope,
)


class RelationalStoreContract:
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

    async def test_satisfies_relational_store_protocol(self, adapters: ContractAdapters) -> None:
        assert isinstance(adapters.relational, RelationalStore)

    # -- Round-trip ---------------------------------------------------------

    async def test_insert_then_get_round_trips(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        await store.insert_memory(make_memory("m1"))
        fetched = await store.get_memory("m1")
        assert fetched is not None
        assert fetched.id == "m1"
        assert fetched.content == "content for m1"

    async def test_get_unknown_id_returns_none(self, adapters: ContractAdapters) -> None:
        assert await adapters.relational.get_memory("missing") is None

    async def test_active_exact_claim_candidate_is_access_scoped_and_excludable(
        self,
        adapters: ContractAdapters,
    ) -> None:
        store = adapters.relational
        exact = make_memory(
            "active-exact",
            content="same active claim",
            project_key="project-a",
            repo_identifier="repo-a",
        )
        different_project = make_memory(
            "active-exact-project-b",
            content="same active claim",
            project_key="project-b",
            repo_identifier="repo-a",
        )
        wrong_repository = make_memory(
            "active-other-repository",
            content="same active claim",
            project_key="project-a",
            repo_identifier="repo-b",
        )
        private_owner_a = make_memory(
            "active-private-owner-a",
            content="same active claim",
            visibility=Visibility.PRIVATE.value,
            owner_user_id="owner-a",
            project_key="project-a",
            repo_identifier="repo-a",
        )
        private_owner_b = make_memory(
            "active-private-owner-b",
            content="same active claim",
            visibility=Visibility.PRIVATE.value,
            owner_user_id="owner-b",
            project_key="project-a",
            repo_identifier="repo-a",
        )
        retired = make_memory(
            "retired-exact",
            content="same active claim",
            status="retired",
            project_key="project-a",
            repo_identifier="repo-a",
        )
        for memory in (
            wrong_repository,
            private_owner_a,
            private_owner_b,
            retired,
            different_project,
            exact,
        ):
            await store.insert_memory(memory)

        candidate = await store.find_active_exact_claim_candidate(
            exact.content_hash,
            visibility=exact.visibility,
            owner_user_id=exact.owner_user_id,
            repo_identifier=exact.repo_identifier,
        )
        excluded = await store.find_active_exact_claim_candidate(
            exact.content_hash,
            visibility=exact.visibility,
            owner_user_id=exact.owner_user_id,
            repo_identifier=exact.repo_identifier,
            excluded_memory_ids=(exact.id,),
        )
        private_candidate = await store.find_active_exact_claim_candidate(
            exact.content_hash,
            visibility=Visibility.PRIVATE.value,
            owner_user_id="owner-a",
            repo_identifier=exact.repo_identifier,
        )
        unknown_private_owner = await store.find_active_exact_claim_candidate(
            exact.content_hash,
            visibility=Visibility.PRIVATE.value,
            owner_user_id="owner-c",
            repo_identifier=exact.repo_identifier,
        )

        assert candidate is not None
        assert candidate.id == exact.id
        assert excluded is not None
        assert excluded.id == different_project.id
        assert private_candidate is not None
        assert private_candidate.id == private_owner_a.id
        assert unknown_private_owner is None

    async def test_active_ordinary_claim_candidates_are_filtered_and_input_ordered(
        self,
        adapters: ContractAdapters,
    ) -> None:
        store = adapters.relational
        first = make_memory("ordinary-first")
        second = make_memory("ordinary-second")
        retired = make_memory("ordinary-retired", status="retired")
        for memory in (second, retired, first):
            await store.insert_memory(memory)

        candidates = await store.list_active_ordinary_claim_memories(
            (first.id, retired.id, "missing", second.id, first.id),
        )

        assert [memory.id for memory in candidates] == [first.id, second.id]

    async def test_active_memory_batch_reads_separate_full_rows_from_candidate_provenance(
        self,
        adapters: ContractAdapters,
    ) -> None:
        store = adapters.relational
        first = make_memory("batch-first", repo_identifier="repo-a")
        second = make_memory(
            "batch-second",
            visibility=Visibility.PRIVATE.value,
            owner_user_id="owner-a",
            repo_identifier="repo-a",
        )
        retired = make_memory("batch-retired", status="retired")
        for memory in (second, retired, first):
            await store.insert_memory(memory)
        for document in (
            make_document("doc-first-a", source="src-a"),
            make_document("doc-first-b", source="src-b"),
            make_document("doc-second", source="src-private"),
        ):
            await store.upsert_document(document)
        await store.add_memory_source(
            first.id,
            "doc-first-a",
            "github",
            None,
            source_updated_at=None,
        )
        await store.add_memory_source(
            first.id,
            "doc-first-b",
            "github",
            None,
            source_updated_at=None,
        )
        await store.add_memory_source(
            second.id,
            "doc-second",
            "jira",
            None,
            source_updated_at=None,
        )

        memories = await store.list_active_memories(
            (first.id, retired.id, "missing", second.id, first.id),
        )
        candidates = await store.list_active_candidate_memories(
            (first.id, retired.id, "missing", second.id, first.id),
        )

        assert [memory.id for memory in memories] == [first.id, second.id]
        assert {
            (
                candidate.memory_id,
                candidate.source_id,
                candidate.doc_id,
                candidate.source_lineage_id,
                candidate.visibility,
                candidate.owner_user_id,
                candidate.repo_identifier,
            )
            for candidate in candidates
        } == {
            (
                first.id,
                "src-a",
                "doc-first-a",
                "doc-first-a",
                Visibility.WORKSPACE.value,
                None,
                "repo-a",
            ),
            (
                first.id,
                "src-b",
                "doc-first-b",
                "doc-first-b",
                Visibility.WORKSPACE.value,
                None,
                "repo-a",
            ),
            (
                second.id,
                "src-private",
                "doc-second",
                "doc-second",
                Visibility.PRIVATE.value,
                "owner-a",
                "repo-a",
            ),
        }

    async def test_rebaseline_reactivation_candidate_is_exact_and_access_scoped(
        self,
        adapters: ContractAdapters,
    ) -> None:
        store = adapters.relational
        exact = replace(
            make_memory("rebaseline-exact", content="same replayed claim", status="retired"),
            retirement_reason="source_rebaseline",
        )
        wrong_reason = replace(
            make_memory("retired-for-another-reason", content="same replayed claim", status="retired"),
            retirement_reason="support_removed",
        )
        wrong_repository = replace(
            make_memory(
                "rebaseline-other-repository",
                content="same replayed claim",
                status="retired",
                repo_identifier="repo-b",
            ),
            retirement_reason="source_rebaseline",
        )
        for memory in (wrong_reason, wrong_repository, exact):
            await store.insert_memory(memory)

        candidate = await store.find_rebaseline_reactivation_candidate(
            exact.content_hash,
            visibility=exact.visibility,
            owner_user_id=exact.owner_user_id,
            repo_identifier=exact.repo_identifier,
        )

        assert candidate is not None
        assert candidate.id == exact.id

    # -- Visibility filtering ----------------------------------------------

    async def test_filter_visible_ids_keeps_only_workspace_rows_by_default(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        await store.insert_memory(make_memory("workspace1"))
        await store.insert_memory(
            make_memory(
                "private1",
                visibility=Visibility.PRIVATE.value,
                owner_user_id="dev",
            )
        )
        # include_private defaults to False, so the private row is hidden
        # even from its own owner.
        visible = await store.filter_visible_ids(["workspace1", "private1"], make_scope())
        assert visible == {"workspace1"}

    async def test_filter_visible_ids_admits_owners_private_rows(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        await store.insert_memory(
            make_memory(
                "private-owner",
                visibility=Visibility.PRIVATE.value,
                owner_user_id="dev",
            )
        )
        await store.insert_memory(
            make_memory(
                "private-other",
                visibility=Visibility.PRIVATE.value,
                owner_user_id="someone-else",
            )
        )
        visible = await store.filter_visible_ids(
            ["private-owner", "private-other"],
            make_scope(include_private=True),
        )
        assert visible == {"private-owner"}

    # -- Lifecycle status filtering ----------------------------------------

    async def test_filter_visible_ids_drops_disallowed_statuses(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        await store.insert_memory(make_memory("active1", status="active"))
        await store.insert_memory(make_memory("retired1", status="retired"))
        visible = await store.filter_visible_ids(["active1", "retired1"], make_scope())
        assert visible == {"active1"}

    async def test_filter_visible_ids_admits_superseded_when_allowed(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        await store.insert_memory(make_memory("active1", status="active"))
        await store.insert_memory(make_memory("supers1", status="superseded"))
        visible = await store.filter_visible_ids(
            ["active1", "supers1"],
            make_scope(statuses=("active", "superseded")),
        )
        assert visible == {"active1", "supers1"}

    # -- Scope mode narrowing ----------------------------------------------

    async def test_project_scope_mode_narrows_to_active_and_shared_keys(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        await store.insert_memory(make_memory("pay1", project_key="PAY"))
        await store.insert_memory(make_memory("risk1", project_key="RISK"))
        await store.insert_memory(make_memory("shared1", project_key=SHARED_PROJECT_KEY))
        await store.insert_memory(make_memory("backlog1", project_key=UNSORTED_PROJECT_KEY))
        visible = await store.filter_visible_ids(
            ["pay1", "risk1", "shared1", "backlog1"],
            make_scope(active_project="PAY", scope_mode="project"),
        )
        assert visible == {"pay1", "shared1"}

    async def test_project_first_scope_mode_keeps_every_project_visible(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        await store.insert_memory(make_memory("pay1", project_key="PAY"))
        await store.insert_memory(make_memory("risk1", project_key="RISK"))
        await store.insert_memory(make_memory("backlog1", project_key=UNSORTED_PROJECT_KEY))
        visible = await store.filter_visible_ids(
            ["pay1", "risk1", "backlog1"],
            make_scope(active_project="PAY", scope_mode="project-first"),
        )
        assert visible == {"pay1", "risk1", "backlog1"}

    # -- Ranking metadata --------------------------------------------------

    async def test_fetch_ranking_metadata_returns_updated_at_and_project(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        await store.insert_memory(make_memory("m1", project_key="PAY"))
        rows = await store.fetch_ranking_metadata(["m1", "missing"])
        assert "missing" not in rows
        meta = rows["m1"]
        assert isinstance(meta["updated_at"], datetime)
        assert meta["project_key"] == "PAY"

    # -- Source facets ------------------------------------------------------

    async def test_filter_ids_by_source_and_time_matches_client_and_repo(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        await store.insert_memory(
            make_memory(
                "m-agent-target",
                repo_identifier="github.tools.sap/hcm/memforge-cloud",
            )
        )
        await store.insert_memory(
            make_memory(
                "m-agent-other-repo",
                repo_identifier="github.tools.sap/hcm/other",
            )
        )
        await store.insert_memory(make_memory("m-jira"))
        await store.upsert_document(make_document("doc-agent-target", client="codex"))
        await store.upsert_document(make_document("doc-agent-other-repo", client="codex"))
        await store.upsert_document(make_document("doc-jira"))
        await store.add_memory_source(
            "m-agent-target",
            "doc-agent-target",
            "agent_session",
            None,
            source_updated_at=None,
        )
        await store.add_memory_source(
            "m-agent-other-repo",
            "doc-agent-other-repo",
            "agent_session",
            None,
            source_updated_at=None,
        )
        await store.add_memory_source("m-jira", "doc-jira", "jira", None, source_updated_at=None)

        matched = await store.filter_ids_by_source_and_time(
            ["m-agent-target", "m-agent-other-repo", "m-jira"],
            MemorySourceFilter(
                clients=("codex",),
                repo_identifiers=("github.tools.sap/hcm/memforge-cloud",),
            ),
            None,
        )

        assert matched == {"m-agent-target"}

    async def test_filter_ids_by_source_and_time_supports_source_and_memory_dates(
        self, adapters: ContractAdapters
    ) -> None:
        store = adapters.relational
        await store.insert_memory(make_memory("m-source-fresh", updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc)))
        await store.insert_memory(make_memory("m-source-stale", updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc)))
        await store.insert_memory(make_memory("m-memory-fresh", updated_at=datetime(2026, 6, 24, tzinfo=timezone.utc)))
        await store.upsert_document(make_document("doc-source-fresh"))
        await store.upsert_document(make_document("doc-source-stale"))
        await store.upsert_document(make_document("doc-memory-fresh"))
        await store.add_memory_source(
            "m-source-fresh",
            "doc-source-fresh",
            "jira",
            None,
            source_updated_at=datetime(2026, 6, 24, 9, 0, tzinfo=timezone.utc),
        )
        await store.add_memory_source(
            "m-source-stale",
            "doc-source-stale",
            "jira",
            None,
            source_updated_at=datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc),
        )
        await store.add_memory_source(
            "m-memory-fresh",
            "doc-memory-fresh",
            "jira",
            None,
            source_updated_at=datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc),
        )

        source_matches = await store.filter_ids_by_source_and_time(
            ["m-source-fresh", "m-source-stale", "m-memory-fresh"],
            None,
            MemoryTimeRange(
                after=datetime(2026, 6, 20, tzinfo=timezone.utc),
                before=datetime(2026, 6, 27, tzinfo=timezone.utc),
                date_type="source_updated_at",
            ),
        )
        memory_matches = await store.filter_ids_by_source_and_time(
            ["m-source-fresh", "m-source-stale", "m-memory-fresh"],
            None,
            MemoryTimeRange(
                after=datetime(2026, 6, 20, tzinfo=timezone.utc),
                before=datetime(2026, 6, 27, tzinfo=timezone.utc),
                date_type="memory_updated_at",
            ),
        )

        assert source_matches == {"m-source-fresh"}
        assert memory_matches == {"m-memory-fresh"}

    async def test_filter_ids_by_source_and_time_scopes_source_date_to_same_provenance_row(
        self,
        adapters: ContractAdapters,
    ) -> None:
        store = adapters.relational
        await store.insert_memory(make_memory("m-mixed"))
        await store.upsert_document(make_document("doc-jira-stale", source="JIRA"))
        await store.upsert_document(make_document("doc-confluence-fresh", source="CONF"))
        await store.add_memory_source(
            "m-mixed",
            "doc-jira-stale",
            "jira",
            None,
            source_updated_at=datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc),
        )
        await store.add_memory_source(
            "m-mixed",
            "doc-confluence-fresh",
            "confluence",
            None,
            source_updated_at=datetime(2026, 6, 24, 9, 0, tzinfo=timezone.utc),
        )

        matched = await store.filter_ids_by_source_and_time(
            ["m-mixed"],
            MemorySourceFilter(source_ids=("JIRA",)),
            MemoryTimeRange(
                after=datetime(2026, 6, 20, tzinfo=timezone.utc),
                before=datetime(2026, 6, 27, tzinfo=timezone.utc),
                date_type="source_updated_at",
            ),
        )

        assert matched == set()

    async def test_filter_ids_by_source_and_time_matches_exact_source_id_on_same_row(
        self,
        adapters: ContractAdapters,
    ) -> None:
        store = adapters.relational
        await store.insert_memory(make_memory("m-target"))
        await store.insert_memory(make_memory("m-other"))
        await store.upsert_document(make_document("doc-target", source="src-mounttai"))
        await store.upsert_document(make_document("doc-other", source="src-other"))
        await store.add_memory_source(
            "m-target",
            "doc-target",
            "jira",
            None,
            source_updated_at=datetime(2026, 6, 24, 9, 0, tzinfo=timezone.utc),
        )
        await store.add_memory_source(
            "m-other",
            "doc-other",
            "jira",
            None,
            source_updated_at=datetime(2026, 6, 24, 9, 0, tzinfo=timezone.utc),
        )

        matched = await store.filter_ids_by_source_and_time(
            ["m-target", "m-other"],
            MemorySourceFilter(source_ids=("src-mounttai",)),
            MemoryTimeRange(
                after=datetime(2026, 6, 20, tzinfo=timezone.utc),
                before=datetime(2026, 6, 27, tzinfo=timezone.utc),
                date_type="source_updated_at",
            ),
        )

        assert matched == {"m-target"}

    async def test_list_ids_by_source_and_time_returns_deterministic_source_date_page(
        self,
        adapters: ContractAdapters,
    ) -> None:
        store = adapters.relational
        await store.insert_memory(make_memory("m-newer", updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc)))
        await store.insert_memory(make_memory("m-older", updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc)))
        await store.insert_memory(make_memory("m-other", updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc)))
        await store.upsert_document(make_document("doc-newer", source="src-mounttai"))
        await store.upsert_document(make_document("doc-older", source="src-mounttai"))
        await store.upsert_document(make_document("doc-other", source="src-other"))
        await store.add_memory_source(
            "m-newer",
            "doc-newer",
            "jira",
            None,
            source_updated_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
        )
        await store.add_memory_source(
            "m-older",
            "doc-older",
            "jira",
            None,
            source_updated_at=datetime(2026, 6, 24, 9, 0, tzinfo=timezone.utc),
        )
        await store.add_memory_source(
            "m-other",
            "doc-other",
            "jira",
            None,
            source_updated_at=datetime(2026, 6, 26, 9, 0, tzinfo=timezone.utc),
        )

        page, total = await store.list_ids_by_source_and_time(
            MemorySourceFilter(source_ids=("src-mounttai",)),
            MemoryTimeRange(
                after=datetime(2026, 6, 20, tzinfo=timezone.utc),
                before=datetime(2026, 6, 27, tzinfo=timezone.utc),
                date_type="source_updated_at",
            ),
            make_scope(),
            limit=1,
            offset=0,
        )

        assert page == ["m-newer"]
        assert total == 2

    async def test_filter_ids_by_source_and_time_does_not_match_created_at(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        await store.insert_memory(
            make_memory(
                "m-created-only",
                created_at=datetime(2026, 6, 24, tzinfo=timezone.utc),
                updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            )
        )

        matched = await store.filter_ids_by_source_and_time(
            ["m-created-only"],
            None,
            MemoryTimeRange(
                after=datetime(2026, 6, 20, tzinfo=timezone.utc),
                before=datetime(2026, 6, 27, tzinfo=timezone.utc),
                date_type="memory_updated_at",
            ),
        )

        assert matched == set()

    async def test_filter_ids_by_source_and_time_memory_date_does_not_require_provenance(
        self,
        adapters: ContractAdapters,
    ) -> None:
        store = adapters.relational
        await store.insert_memory(
            make_memory(
                "m-unbacked-fresh",
                updated_at=datetime(2026, 6, 24, tzinfo=timezone.utc),
            )
        )

        matched = await store.filter_ids_by_source_and_time(
            ["m-unbacked-fresh"],
            None,
            MemoryTimeRange(
                after=datetime(2026, 6, 20, tzinfo=timezone.utc),
                before=datetime(2026, 6, 27, tzinfo=timezone.utc),
                date_type="memory_updated_at",
            ),
        )

        assert matched == {"m-unbacked-fresh"}

    async def test_filter_ids_by_source_and_time_source_date_requires_provenance(
        self,
        adapters: ContractAdapters,
    ) -> None:
        store = adapters.relational
        await store.insert_memory(
            make_memory(
                "m-unbacked-fresh",
                updated_at=datetime(2026, 6, 24, tzinfo=timezone.utc),
            )
        )

        matched = await store.filter_ids_by_source_and_time(
            ["m-unbacked-fresh"],
            None,
            MemoryTimeRange(
                after=datetime(2026, 6, 20, tzinfo=timezone.utc),
                before=datetime(2026, 6, 27, tzinfo=timezone.utc),
                date_type="source_updated_at",
            ),
        )

        assert matched == set()

    # -- Project lifecycle --------------------------------------------------

    async def test_project_create_and_get_round_trips(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        created = await store.create_project(key="PAY", name="Payments")
        fetched = await store.get_project(created.id)
        assert fetched is not None
        assert fetched.key == "PAY"
        assert fetched.name == "Payments"

    async def test_list_projects_returns_every_known_row(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        await store.create_project(key="PAY", name="Payments")
        await store.create_project(key="RISK", name="Risk")
        keys = {p.key for p in await store.list_projects()}
        assert {"PAY", "RISK"}.issubset(keys)

    async def test_update_project_renames_in_place(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        created = await store.create_project(key="PAY", name="Pay v1")
        updated = await store.update_project(created.id, name="Pay v2")
        assert updated is not None
        assert updated.name == "Pay v2"

    async def test_list_project_memory_ids_returns_attached_rows(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        project = await store.create_project(key="PAY", name="Payments")
        await store.insert_memory(make_memory("m-pay", project_key="PAY"))
        await store.insert_memory(make_memory("m-other", project_key="RISK"))
        ids = await store.list_project_memory_ids(project.id)
        assert ids == ["m-pay"]

    async def test_list_project_memory_ids_rejects_unknown_project(self, adapters: ContractAdapters) -> None:
        store = adapters.relational
        with pytest.raises(LookupError):
            await store.list_project_memory_ids("proj-does-not-exist")

    async def test_commit_project_deletion_rebuckets_to_unsorted_and_drops_row(
        self, adapters: ContractAdapters
    ) -> None:
        store = adapters.relational
        project = await store.create_project(key="PAY", name="Payments")
        await store.insert_memory(make_memory("m-pay", project_key="PAY"))
        affected = await store.list_project_memory_ids(project.id)
        await store.commit_project_deletion(project.id, affected)

        # The project row is gone.
        assert await store.get_project(project.id) is None
        # The named row was rebucketed; the relational and vector channels
        # share the affected list, so this exact set must be the rebucketed
        # set.
        rebucketed = await store.get_memory("m-pay")
        assert rebucketed is not None
        assert rebucketed.project_key == UNSORTED_PROJECT_KEY
