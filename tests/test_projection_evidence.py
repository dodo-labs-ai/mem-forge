from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from memforge.models import ContentItem, NormalizedContent, RawContent, RawMemory, content_hash
from memforge.pipeline.projection_evidence import build_projected_claim_evidence
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_artifacts import StoredSourceArtifact
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
        "fields": {
            "summary": "Payroll",
            "description": "Issue context",
            "status": None,
            "priority": None,
            "assignee": None,
            "labels": [],
            "resolution": None,
            "updated": "2026-07-15T00:00:00Z",
        },
        "_comments": [
            {"id": "501", "body": "Question", "created": "2026-07-15T10:00:00Z"},
            {
                "id": "502",
                "body": "Correction: retain A7",
                "created": "2026-07-15T10:01:00Z",
            },
        ],
        "_comments_included": True,
        "_comments_total": 2,
        "changelog": {"startAt": 0, "histories": [], "total": 0},
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


def test_visual_claim_is_bound_to_exact_artifact_revision_without_fabricated_quote() -> None:
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
        "fields": {
            "summary": "Payroll",
            "description": "Issue context",
            "status": None,
            "priority": None,
            "assignee": None,
            "labels": [],
            "resolution": None,
            "updated": "2026-07-15T00:00:00Z",
        },
        "_comments": [
            {"id": "502", "body": "See result", "created": "2026-07-15T10:01:00Z"}
        ],
        "_comments_included": True,
        "_comments_total": 1,
        "changelog": {"startAt": 0, "histories": [], "total": 0},
    }
    artifact = StoredSourceArtifact(
        id="artifact-1",
        provider_key="attachment-1",
        parent_observation_type="comment",
        parent_provider_key="502",
        provider_revision="1",
        filename="result.png",
        media_type="image/png",
        size_bytes=4,
        sha256="a" * 64,
        uri="/stored/result.png",
    )
    projection = project_source_item(
        source_id="src-jira",
        source_type="jira",
        run_id="projection-jira-image",
        item=item,
        raw=RawContent(
            item=item,
            body=json.dumps(payload).encode(),
            content_type="application/json",
        ),
        normalized=NormalizedContent(
            item=item,
            markdown_body="# PAY-12\n\nIssue context\n\nSee result",
        ),
        artifacts=(artifact,),
    )
    artifact_observation = next(
        item for item in projection.observations if item.observation_type == "binary_artifact"
    )
    artifact_revision = next(
        item
        for item in projection.observation_revisions
        if item.observation_id == artifact_observation.id
    )
    raw = RawMemory(
        content="The screenshot shows a settled validation result.",
        memory_type="fact",
        source_observation_id=artifact_observation.id,
        evidence_anchor="source_artifact",
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

    primary = next(item for item in staged.references if item.role.value == "primary")
    assert primary.anchor.observation_revision_id == artifact_revision.id
    assert staged.units[0].content == ""
    assert staged.units[0].excerpt is None
    assert staged.units[0].evidence_provenance.value == "source_artifact"


@pytest.mark.asyncio
async def test_jira_comment_can_promote_description_to_required_evidence(db: Database) -> None:
    projection = _jira_projection()
    primary_id = projection.observations[2].id
    description_id = projection.observations[0].id
    raw = RawMemory(
        content="A7 is retained for this issue context.",
        memory_type="decision",
        evidence_quote="Correction: retain A7",
        source_observation_id=primary_id,
        required_source_observation_ids=[description_id],
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
        extractor_run_id="sync-required",
    )

    references = staged.references
    assert [(item.role.value, item.anchor.observation_id) for item in references] == [
        ("primary", primary_id),
        ("required", description_id),
        ("context", projection.observations[1].id),
    ]
    assert len(staged.reference_ids_by_claim_hash[content_hash(raw.content)]) == 2


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
async def test_out_of_scope_observation_hint_rebinds_to_unique_current_quote(
    db: Database,
) -> None:
    projection = _jira_projection()
    quote = "Correction: retain A7"
    raw = RawMemory(
        content="The correction retains A7.",
        memory_type="decision",
        evidence_quote=quote,
        source_observation_id="obs-from-another-projection",
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
        extractor_run_id="sync-rebind",
    )

    [primary] = [item for item in staged.references if item.role.value == "primary"]
    assert primary.anchor.observation_id == projection.observations[2].id


@pytest.mark.asyncio
async def test_added_observation_is_in_current_evidence_scope_with_other_changed_observations(
    db: Database,
) -> None:
    projection = _jira_projection()
    target_id = projection.observations[2].id
    other_id = projection.observations[1].id
    delta = projection.deltas[0]
    projection = replace(
        projection,
        deltas=(
            replace(
                delta,
                changed_anchors=tuple(
                    anchor for anchor in delta.changed_anchors if anchor.observation_id == other_id
                ),
                added_observation_ids=(target_id,),
            ),
        ),
    )
    raw = RawMemory(
        content="The correction retains A7.",
        memory_type="decision",
        evidence_quote="Correction: retain A7",
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
        extractor_run_id="sync-added-primary",
    )

    [primary] = [item for item in staged.references if item.role.value == "primary"]
    assert primary.anchor.observation_id == target_id


@pytest.mark.asyncio
async def test_out_of_scope_observation_hint_without_unique_quote_is_rejected(
    db: Database,
) -> None:
    projection = _jira_projection()
    raw = RawMemory(
        content="The correction retains A7.",
        memory_type="decision",
        evidence_quote="missing from the current projection",
        source_observation_id="obs-from-another-projection",
    )

    with pytest.raises(ValueError, match="outside the current evidence scope"):
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
            extractor_run_id="sync-reject",
        )


@pytest.mark.asyncio
async def test_out_of_scope_observation_hint_with_ambiguous_quote_is_rejected(
    db: Database,
) -> None:
    projection = _jira_projection()
    repeated_quote = "Correction: retain A7"
    duplicate_target_id = projection.observations[1].id
    projection = replace(
        projection,
        observation_revisions=tuple(
            replace(revision, content=f"{revision.content}\n{repeated_quote}")
            if revision.observation_id == duplicate_target_id
            else revision
            for revision in projection.observation_revisions
        ),
    )
    raw = RawMemory(
        content="The correction retains A7.",
        memory_type="decision",
        evidence_quote=repeated_quote,
        source_observation_id="obs-from-another-projection",
    )

    with pytest.raises(ValueError, match="outside the current evidence scope"):
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
            extractor_run_id="sync-ambiguous",
        )


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
