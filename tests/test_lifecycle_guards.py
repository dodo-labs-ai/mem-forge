"""Same-Unit identity backstop and DestructiveValidation end to end, on SQLite."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from memforge.llm.structured import MemoryRelationCatalogResponse
from memforge.memory.engine import MemoryEngine
from memforge.models import Memory, RawMemory
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database
from tests.test_projected_lifecycle_integration import (
    db as db,
    _OutboxDrainer,
    _candidate_retriever,
    _jira_projection,
    _selected,
    _set_fixture_source_type,
)
from tests.test_support_relation_coordination import (
    RESTATED,
    TWO,
    _ScriptedClient,
    _seeded_page,
    _support_texts,
)
from tests.unit_support_fixture import active_support_evidence


class _IdentityClient(_ScriptedClient):
    """Relation finds no edge, while identity judges the scripted pairs equivalent."""

    def __init__(self, *, identity: set[tuple[str, str]], **kwargs) -> None:
        super().__init__(**kwargs)
        self.identity = identity

    async def discover_memory_relations(self, prompt, **kwargs):
        payload = json.loads(prompt.split("<memory_relation_catalog>\n", 1)[1].split(
            "\n</memory_relation_catalog>", 1)[0])
        old = {row["id"]: row["content"] for row in payload["existing_claims"]}
        return MemoryRelationCatalogResponse.model_validate(dict(results=[
            dict(candidate_id=row["id"], relations=[
                dict(existing_id=ref, classification="equivalent", direction="symmetric",
                     same_subject_and_scope=True, incompatible_assertions="", reason="fixture identity")
                for ref in payload["allowed_existing_ids"][row["id"]] if (row["content"], old[ref]) in self.identity
            ])
            for row in payload["new_claims"]
        ]))


class _IdentityStore(_OutboxDrainer):
    """Offers one Memory as the equivalence candidate unless identity excludes it."""

    def __init__(self, database: Database, target: Memory) -> None:
        super().__init__(database)
        self.target = target
        self.exclusions: list[frozenset[str]] = []

    async def find_access_compatible_equivalence_candidates(self, memory, *, excluded_memory_ids=frozenset(), **kwargs):
        self.exclusions.append(frozenset(excluded_memory_ids))
        if self.target.id in excluded_memory_ids:
            return ()
        return (await self.db.get_memory(self.target.id),)


def _engine(db: Database, client, memory_store=None) -> MemoryEngine:
    return MemoryEngine(
        cross_document_candidates=_candidate_retriever(build_sqlite_adapters(db, object())),
        db=db,
        memory_store=memory_store or _OutboxDrainer(db),
        structured_llm_client=client,
    )


@pytest.mark.asyncio
async def test_identity_attaches_an_equivalent_that_relation_missed_to_the_kept_old_memory(db: Database) -> None:
    page, memory = await _seeded_page(db, TWO)
    store = _IdentityStore(db, memory)
    projection = page.next(TWO, RESTATED)

    stats = await _engine(db, _IdentityClient(identity={(RESTATED, TWO)}), store).prepare_and_commit_projected_lifecycle(
        projection=projection,
        doc_id="confluence-123",
        raw_memories=_selected(projection, [RawMemory(content=RESTATED, memory_type="fact")]),
        doc_type="design-doc", project_key="ENG", repo_identifier=None,
        document_content=projection.observation_revisions[-1].content,
        update_mode="full_document", changed_hunks=None, update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
    )

    # The rebound old Memory is an eligible identity target, so no duplicate is created.
    assert memory.id not in store.exclusions[0]
    assert stats["added"] == 0 and stats["corroborated"] == 1
    assert [item.id for item in await db.list_memories() if item.status == "active"] == [memory.id]
    assert await _support_texts(db, memory.id) == {TWO, RESTATED}


async def _jira_page(db: Database):
    """A description claim and a comment claim; the next, partial, page of comments leaves the comment out."""
    await _set_fixture_source_type(db, "jira")
    await db.enable_lifecycle_gate("src-1")
    first = _jira_projection(
        run_id="projection-jira-guard-1", description="Payroll runs weekly.",
        comment_id="501", comment_body="Decision: retain A7",
    )
    await _engine(db, _ScriptedClient()).prepare_and_commit_projected_lifecycle(
        projection=first, doc_id="confluence-123",
        raw_memories=_selected(first, [
            RawMemory(content="Payroll runs weekly.", memory_type="fact"),
            RawMemory(content="A7 is retained.", memory_type="decision", evidence_quote="Decision: retain A7"),
        ]),
        doc_type="ticket", project_key="ENG", repo_identifier=None, document_content="PAY-12",
        update_mode="diff_guided", changed_hunks=None, update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    memories = {memory.content: memory for memory in await db.list_memories()}
    second = _jira_projection(
        run_id="projection-jira-guard-2", description="",
        comment_id="502", comment_body="Unrelated follow-up.", comments_truncated=True,
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    assert second.coverage.value == "partial_projection"
    return memories["Payroll runs weekly."], memories["A7 is retained."], second


async def _commit(db: Database, projection, *, document_content: str):
    return await _engine(db, _ScriptedClient()).prepare_and_commit_projected_lifecycle(
        projection=projection, doc_id="confluence-123", raw_memories=[],
        doc_type="ticket", project_key="ENG", repo_identifier=None, document_content=document_content,
        update_mode="diff_guided", changed_hunks=None, update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("document_content", ["PAY-12", ""], ids=["read", "explicitly-empty"])
async def test_partial_projection_removes_support_only_where_the_returned_content_is_authoritative(
    db: Database, document_content: str,
) -> None:
    weekly, retained, second = await _jira_page(db)
    comment_support = await active_support_evidence(db, retained.id, source_id="src-1")

    stats = await _commit(db, second, document_content=document_content)

    # The description was returned without the sentence: its claim loses its only Support.
    assert (await db.get_memory(weekly.id)).status == "retired"
    assert stats["deleted"] == 1
    # The comment was not returned, so absence is unknown and its claim keeps its Support.
    assert (await db.get_memory(retained.id)).status == "active"
    assert await active_support_evidence(db, retained.id, source_id="src-1") == comment_support
    # Reading routes an UNKNOWN part to UNRESOLVED before any decision; the empty
    # revision has nothing to read, so DestructiveValidation keeps the claim.
    expected_kept = 1 if not document_content else 0
    assert stats["destructive_validation_kept_support_unresolved_count"] == expected_kept
