from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from memforge.models import ContentItem, NormalizedContent, RawContent, RawMemory, content_hash
from memforge.pipeline.projection_evidence import build_projected_claim_evidence
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.storage.database import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "projection-evidence.db"))
    await database.connect()
    await database.upsert_source(
        id="src-jira",
        type="jira",
        name="Jira",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    try:
        yield database
    finally:
        await database.close()


def _jira_projection():
    item = ContentItem(
        item_id="jira-PAY-12",
        title="PAY-12",
        source_url="https://jira.example/browse/PAY-12",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
        version="2",
        extra={"issue_key": "PAY-12", "issue_id": "10012"},
    )
    payload = {
        "id": "10012",
        "key": "PAY-12",
        "fields": {"summary": "Payroll", "description": "Issue context"},
        "_comments": [
            {"id": "501", "body": "Question", "created": "2026-07-15T10:00:00Z"},
            {
                "id": "502",
                "body": "Correction: retain A7",
                "created": "2026-07-15T10:01:00Z",
            },
        ],
    }
    raw = RawContent(
        item=item,
        body=json.dumps(payload).encode(),
        content_type="application/json",
    )
    normalized = NormalizedContent(
        item=item,
        markdown_body="# PAY-12\n\nIssue context\n\nCorrection: retain A7",
    )
    return project_source_item(
        source_id="src-jira",
        source_type="jira",
        run_id="projection-jira-1",
        item=item,
        raw=raw,
        normalized=normalized,
    )


@pytest.mark.asyncio
async def test_short_jira_comment_is_primary_with_adjacent_context_only(db: Database) -> None:
    projection = _jira_projection()
    await db.record_source_projection(projection)
    raw = RawMemory(
        content="A7 is retained.",
        memory_type="decision",
        extraction_context="Correction: retain A7",
    )

    staged = build_projected_claim_evidence(
        projection=projection,
        raw_memories=(raw,),
        doc_id="jira-PAY-12",
        source_type="jira",
        project_key="PAY",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash="workspace-pay",
        extractor_run_id="sync-1",
    )

    assert len(staged.reference_ids_by_claim_hash[content_hash(raw.content)]) == 1
    primary = [item for item in staged.references if item.role.value == "primary"]
    context = [item for item in staged.references if item.role.value == "context"]
    assert len(primary) == 1
    assert primary[0].anchor.observation_id == projection.observations[2].id
    assert {item.anchor.observation_id for item in context} == {
        projection.observations[0].id,
        projection.observations[1].id,
    }
    assert await db.db.execute_fetchall("SELECT id FROM evidence_units") == []
    assert await db.db.execute_fetchall("SELECT id FROM evidence_references") == []


@pytest.mark.asyncio
async def test_ambiguous_multi_observation_claim_is_rejected(db: Database) -> None:
    projection = _jira_projection()
    await db.record_source_projection(projection)
    raw = RawMemory(content="A7 is retained.", memory_type="decision")

    with pytest.raises(ValueError, match="cannot be localized"):
        build_projected_claim_evidence(
            projection=projection,
            raw_memories=(raw,),
            doc_id="jira-PAY-12",
            source_type="jira",
            project_key="PAY",
            visibility="workspace",
            owner_user_id=None,
            repo_identifier=None,
            access_context_hash="workspace-pay",
            extractor_run_id="sync-1",
        )


@pytest.mark.asyncio
async def test_explicit_source_observation_disambiguates_repeated_quote(db: Database) -> None:
    projection = _jira_projection()
    await db.record_source_projection(projection)
    target_id = projection.observations[2].id
    raw = RawMemory(
        content="The correction retains A7.",
        memory_type="decision",
        evidence_quote="retain A7",
        source_observation_id=target_id,
    )

    staged = build_projected_claim_evidence(
        projection=projection,
        raw_memories=(raw,),
        doc_id="jira-PAY-12",
        source_type="jira",
        project_key="PAY",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash="workspace-pay",
        extractor_run_id="sync-1",
    )

    primary = [item for item in staged.references if item.role.value == "primary"]
    assert len(primary) == 1
    assert primary[0].anchor.observation_id == target_id


@pytest.mark.asyncio
async def test_explicit_source_observation_must_contain_evidence_quote(db: Database) -> None:
    projection = _jira_projection()
    await db.record_source_projection(projection)
    raw = RawMemory(
        content="The correction retains A7.",
        memory_type="decision",
        evidence_quote="Correction: retain A7",
        source_observation_id=projection.observations[1].id,
    )

    with pytest.raises(ValueError, match="does not contain the evidence quote"):
        build_projected_claim_evidence(
            projection=projection,
            raw_memories=(raw,),
            doc_id="jira-PAY-12",
            source_type="jira",
            project_key="PAY",
            visibility="workspace",
            owner_user_id=None,
            repo_identifier=None,
            access_context_hash="workspace-pay",
            extractor_run_id="sync-1",
        )
