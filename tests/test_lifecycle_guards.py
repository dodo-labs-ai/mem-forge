"""DestructiveValidation end to end, on SQLite."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.models import RawMemory
from memforge.storage.database import Database
from tests.coordination_fixture import (
    DOC_ID,
    JIRA_DOCUMENT,
    SOURCE_ID,
    ScriptedClient,
    coordination_engine,
    partial_jira_revision,
    seed_jira_issue,
)
from tests.test_projected_lifecycle_integration import db as db
from tests.unit_support_fixture import active_support_evidence


async def _jira_page(db: Database):
    """A description claim and a comment claim; the next, partial, page of comments leaves the comment out."""
    first = await seed_jira_issue(db, run_id="projection-jira-guard-1", description="Payroll runs weekly.", claims=[
        RawMemory(content="Payroll runs weekly.", memory_type="fact"),
        RawMemory(content="A7 is retained.", memory_type="decision", evidence_quote="Decision: retain A7"),
    ])
    memories = {memory.content: memory for memory in await db.list_memories()}
    second = partial_jira_revision(first, run_id="projection-jira-guard-2", description="")
    return memories["Payroll runs weekly."], memories["A7 is retained."], second


async def _commit(db: Database, projection, *, document_content: str):
    return await coordination_engine(db, ScriptedClient()).prepare_and_commit_projected_lifecycle(
        projection=projection, doc_id=DOC_ID, raw_memories=[],
        doc_type="ticket", project_key="ENG", repo_identifier=None, document_content=document_content,
        update_mode="diff_guided", changed_hunks=None, update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("document_content", [JIRA_DOCUMENT, ""], ids=["read", "explicitly-empty"])
async def test_partial_projection_removes_support_only_where_the_returned_content_is_authoritative(
    db: Database, document_content: str,
) -> None:
    weekly, retained, second = await _jira_page(db)
    comment_support = await active_support_evidence(db, retained.id, source_id=SOURCE_ID)

    stats = await _commit(db, second, document_content=document_content)

    # The description was returned without the sentence: its claim loses its only Support.
    assert (await db.get_memory(weekly.id)).status == "retired"
    assert stats["deleted"] == 1
    # The comment was not returned, so absence is unknown and its claim keeps its Support.
    assert (await db.get_memory(retained.id)).status == "active"
    assert await active_support_evidence(db, retained.id, source_id=SOURCE_ID) == comment_support
    # Reading routes an UNKNOWN part to UNRESOLVED before any decision; the empty
    # revision has nothing to read, so DestructiveValidation keeps the claim.
    expected_kept = 1 if not document_content else 0
    assert stats["destructive_validation_kept_support_unresolved_count"] == expected_kept
