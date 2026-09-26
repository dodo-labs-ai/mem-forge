from __future__ import annotations

from dataclasses import replace
import json
from datetime import datetime, timezone

import pytest

from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.llm.structured import StructuredLlmImage
from memforge.pipeline.extraction_requests import plan_extraction_requests
from memforge.pipeline.projection_context import (
    CommittedSourceUnitSnapshot,
    ExtractionAuthority,
    plan_projection_evidence_work,
    preceding_observation_id,
)
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_projection import (
    with_source_artifact_summaries,
)
from memforge.source_artifacts import (
    SourceArtifactSummary,
    StoredSourceArtifact,
)
from memforge.source_representation import UNIT_TITLE_OBSERVATION_TYPE
from tests.llm_fixture import NoopMemoryExtractor


def _committed_snapshot(projection) -> CommittedSourceUnitSnapshot:
    return CommittedSourceUnitSnapshot(
        unit_revision=projection.source_unit_revisions[0],
        observation_revisions=projection.observation_revisions,
    )


def _jira_projection(comment_count: int = 3):
    item = ContentItem(
        item_id="jira-PAY-12",
        title="Payroll",
        source_url="https://jira.example.test/browse/PAY-12",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
        version="2",
        extra={"issue_key": "PAY-12"},
    )
    payload = {
        "id": "10012",
        "key": "PAY-12",
        "fields": {
            "summary": "Payroll",
            "description": "A7 processing context",
            "status": None,
            "priority": None,
            "assignee": None,
            "labels": [],
            "resolution": None,
            "updated": "2026-07-15T00:00:00Z",
        },
        "_comments": [
            {"id": str(500 + index), "body": f"Reply {index}: retain A7"}
            for index in range(comment_count)
        ],
        "_comments_included": True,
        "_comments_total": comment_count,
        "changelog": {"startAt": 0, "histories": [], "total": 0},
    }
    import json

    return project_source_item(
        source_id="src-j",
        source_type="jira",
        run_id="run-j",
        item=item,
        raw=RawContent(item=item, body=json.dumps(payload).encode(), content_type="application/json"),
        normalized=NormalizedContent(item=item, markdown_body="normalized Jira"),
    )


def _confluence_projection(body: str):
    item = ContentItem(
        item_id="confluence-42",
        title="Large design",
        source_url="https://confluence.example.test/pages/42",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
        version="7",
        extra={"page_id": "42", "space_key": "ENG"},
    )
    return project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-c",
        item=item,
        raw=RawContent(item=item, body=body.encode(), content_type="text/html"),
        normalized=NormalizedContent(item=item, markdown_body=body),
    )


def _confluence_projection_with_images(
    image_count: int,
    *,
    artifact_size: int = 10,
    inference_eligible: bool = True,
    media_type: str = "image/png",
):
    item = ContentItem(
        item_id="confluence-42",
        title="Visual design",
        source_url="https://confluence.example.test/pages/42",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
        version="7",
        extra={"page_id": "42", "space_key": "ENG"},
    )
    artifacts = tuple(
        StoredSourceArtifact(
            id=f"artifact-{index}",
            provider_key=f"attachment-{index}",
            parent_observation_type="page_body",
            parent_provider_key="42:body",
            provider_revision="1",
            filename=f"diagram-{index}.png",
            media_type=media_type,
            size_bytes=artifact_size,
            sha256=f"{index:064x}",
                uri=f"source-artifacts/src-c/artifact-{index}.png",
                inference_eligible=inference_eligible,
                inference_ineligible_reason=(
                    None
                    if inference_eligible
                    else "invalid_image_structure"
                ),
            )
        for index in range(image_count)
    )
    return project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-c",
        item=item,
        raw=RawContent(item=item, body=b"<p>Visual design.</p>", content_type="text/html"),
        normalized=NormalizedContent(item=item, markdown_body="Visual design."),
        artifacts=artifacts,
    )


def _teams_projection(
    message_contents: tuple[str, str] = (
        "Decision: keep A7",
        "Acknowledged: keep A7",
    ),
    *,
    run_id: str = "run-teams",
    prior_unit_revision=None,
    prior_observation_revisions=None,
):
    item = ContentItem(
        item_id="teams-window-1",
        title="PCC Agent Dev",
        source_url="https://teams.example.test/conversations/conv-1",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
        version="2",
        extra={
            "conversation_id": "conv-1",
            "window_id": "window-1",
            "root_message_id": "msg-1",
        },
    )
    import json

    messages = {
        "messages": [
            {
                "id": "msg-1",
                "content": message_contents[0],
                "time": "2026-07-15T10:00:00Z",
            },
            {
                "id": "msg-2",
                "content": message_contents[1],
                "time": "2026-07-15T10:01:00Z",
            },
        ]
    }
    return project_source_item(
        source_id="src-teams",
        source_type="teams",
        run_id=run_id,
        item=item,
        raw=RawContent(item=item, body=json.dumps(messages).encode(), content_type="application/json"),
        normalized=NormalizedContent(item=item, markdown_body="normalized Teams window"),
        prior_unit_revision=prior_unit_revision,
        prior_observation_revisions=prior_observation_revisions,
    )


def _requests(projection, *, base=None, committed=None, reprocess=False, images=()):
    """Plan the authority, then the production extraction requests, of one projection."""
    authority = plan_projection_evidence_work(
        projection,
        committed_base_snapshot=committed,
        reprocess_all_current_observations=reprocess,
    )
    assert isinstance(authority, ExtractionAuthority)
    return plan_extraction_requests(
        RevisionAssessmentContext(projection=projection, base=base, access_context_hash="scope", images=images),
        authority,
        extractor=NoopMemoryExtractor(),
        source_type=projection.source_type,
        doc_type="document",
    ).requests


def _primary(requests):
    return [fragment for request in requests for fragment in request.catalog.fragments if fragment.primary_eligible]


def _provider_revision(projection):
    """The revision of the provider's first Observation; the Unit Title precedes it."""
    observation_id = next(
        item.id for item in projection.observations if item.observation_type != UNIT_TITLE_OBSERVATION_TYPE
    )
    return next(item for item in projection.observation_revisions if item.observation_id == observation_id)


def _artifact_images(projection):
    return tuple(
        StructuredLlmImage(source_observation_id=item.id, media_type="image/png", body=b"image")
        for item in projection.observations
        if item.observation_type == "binary_artifact"
    )


def test_v9_incremental_markdown_authorizes_only_changed_complete_structures() -> None:
    initial_body = """# Historical decision

Keep the existing approval rule.

# Current update

The rollout starts on Monday.
"""
    target_body = initial_body.replace(
        "The rollout starts on Monday.",
        "The rollout starts on Tuesday.",
    )
    initial = _confluence_projection(initial_body)
    target = project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-c-incremental",
        item=replace(
            ContentItem(
                item_id="confluence-42",
                title="Large design",
                source_url="https://confluence.example.test/pages/42",
                last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
                version="7",
                extra={"page_id": "42", "space_key": "ENG"},
            ),
            version="8",
        ),
        raw=RawContent(
            item=ContentItem(
                item_id="confluence-42",
                title="Large design",
                source_url="https://confluence.example.test/pages/42",
                last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
                version="8",
                extra={"page_id": "42", "space_key": "ENG"},
            ),
            body=target_body.encode(),
            content_type="text/html",
        ),
        normalized=NormalizedContent(
            item=ContentItem(
                item_id="confluence-42",
                title="Large design",
                source_url="https://confluence.example.test/pages/42",
                last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
                version="8",
                extra={"page_id": "42", "space_key": "ENG"},
            ),
            markdown_body=target_body,
        ),
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision
            for revision in initial.observation_revisions
        },
    )
    primary_text = "\n".join(
        fragment.presentation_text
        for fragment in _primary(_requests(target, base=initial, committed=_committed_snapshot(initial)))
    )

    assert "The rollout starts on Tuesday." in primary_text
    assert "Keep the existing approval rule." not in primary_text


def test_v9_incremental_canonical_record_authorizes_only_changed_registered_field() -> None:
    import json

    initial = _jira_projection(0)
    item = ContentItem(
        item_id="jira-PAY-12",
        title="Payroll",
        source_url="https://jira.example.test/browse/PAY-12",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
        version="3",
        extra={"issue_key": "PAY-12"},
    )
    payload = {
        "id": "10012",
        "key": "PAY-12",
        "fields": {
            "summary": "Payroll",
            "description": "A7 processing requires approval",
            "status": None,
            "priority": None,
            "assignee": None,
            "labels": [],
            "resolution": None,
            "updated": "2026-07-16T00:00:00Z",
        },
        "_comments": [],
        "_comments_included": True,
        "_comments_total": 0,
        "changelog": {"startAt": 0, "histories": [], "total": 0},
    }
    target = project_source_item(
        source_id="src-j",
        source_type="jira",
        run_id="run-j-field-update",
        item=item,
        raw=RawContent(
            item=item,
            body=json.dumps(payload).encode(),
            content_type="application/json",
        ),
        normalized=NormalizedContent(item=item, markdown_body="normalized Jira"),
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision
            for revision in initial.observation_revisions
        },
    )

    primary_text = [
        fragment.presentation_text
        for fragment in _primary(_requests(target, base=initial, committed=_committed_snapshot(initial)))
    ]

    assert primary_text == ["A7 processing requires approval"]


@pytest.mark.parametrize("changed", ["Rollout starts Tuesday.", "| Rule | Result |\n| --- | --- |\n" + "| approval | required |\n" * 2_000])
def test_v9_incremental_nested_canonical_text_keeps_unchanged_paragraph_non_primary(changed) -> None:
    initial = _teams_projection(
        (
            "Historical context remains.\n\nRollout starts Monday.",
            "Independent message.",
        )
    )
    target = _teams_projection(
        (
            "Historical context remains.\n\n" + changed,
            "Independent message.",
        ),
        run_id="run-teams-nested-update",
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision
            for revision in initial.observation_revisions
        },
    )

    primary = _primary(_requests(target, base=initial, committed=_committed_snapshot(initial)))
    primary_text = "\n".join(fragment.presentation_text for fragment in primary)

    assert changed.strip() in primary_text
    assert "Historical context remains." not in primary_text
    assert len(primary) == 1
    assert primary[0].presentation_text.strip() == changed.strip()


@pytest.mark.parametrize("reprocess", [False, True])
def test_v9_initial_tombstoned_message_has_no_primary_work(reprocess) -> None:
    import json

    item = ContentItem(
        item_id="teams-window-deleted",
        title="Deleted window",
        source_url="https://teams.example.test/conversations/conv-1",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
        version="1",
        extra={
            "conversation_id": "conv-1",
            "window_id": "window-deleted",
            "root_message_id": "msg-deleted",
        },
    )
    payload = {
        "messages": [
            {
                "id": "msg-deleted",
                "content": "Obsolete decision.",
                "deletedDateTime": "2026-07-15T10:00:00Z",
                "time": "2026-07-15T09:00:00Z",
            }
        ]
    }
    projection = project_source_item(
        source_id="src-teams",
        source_type="teams",
        run_id="run-teams-deleted-initial",
        item=item,
        raw=RawContent(
            item=item,
            body=json.dumps(payload).encode(),
            content_type="application/json",
        ),
        normalized=NormalizedContent(
            item=item,
            markdown_body="normalized Teams window",
        ),
    )

    authority = plan_projection_evidence_work(
        projection,
        reprocess_all_current_observations=reprocess,
    )

    assert isinstance(authority, ExtractionAuthority)
    assert set(authority.ranges_by_observation_id) == {projection.observations[0].id}
    assert _requests(projection, reprocess=reprocess) == ()


def test_v9_batches_keep_one_crossing_markdown_structure_complete() -> None:
    prefix = ("Short paragraph.\n\n" * 3_100)
    code_block = "```text\n" + ("x" * 12_000) + "\n```\n"
    body = prefix + code_block
    projection = _confluence_projection(body)

    catalogs = [request.catalog for request in _requests(projection)]

    assert all(catalog.usable for catalog in catalogs)
    revision_content = _provider_revision(projection).content
    code_start = revision_content.index("```text")
    code_end = len(revision_content)
    code_fragments = [
        fragment
        for catalog in catalogs
        for fragment in catalog.fragments
        if fragment.fragment_type == "markdown-code-block"
    ]
    assert len(code_fragments) == 1
    assert code_fragments[0].anchor.range_start == code_start
    assert code_fragments[0].anchor.range_end == code_end


def test_v9_incremental_oversized_structure_keeps_only_changed_primary_authority() -> None:
    initial_body = "# Decision\n\n```text\nsmall\n```\n"
    target_body = "# Decision\n\n```text\n" + ("x" * 90_000) + "\n```\n"
    initial = _confluence_projection(initial_body)
    target_item = ContentItem(
        item_id="confluence-42",
        title="Large design",
        source_url="https://confluence.example.test/pages/42",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
        version="8",
        extra={"page_id": "42", "space_key": "ENG"},
    )
    target = project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-c-oversized-incremental",
        item=target_item,
        raw=RawContent(
            item=target_item,
            body=target_body.encode(),
            content_type="text/html",
        ),
        normalized=NormalizedContent(item=target_item, markdown_body=target_body),
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision
            for revision in initial.observation_revisions
        },
    )

    [request] = _requests(target, base=initial, committed=_committed_snapshot(initial))
    catalog = request.catalog
    assert catalog.usable
    primary = [f for f in catalog.fragments if f.primary_eligible]
    assert len(primary) == 1
    assert primary[0].presentation_text == "```text\n" + ("x" * 90_000) + "\n```"
    assert not any(f.primary_eligible and "# Decision" in f.presentation_text for f in catalog.fragments)


def test_v9_structure_planning_keeps_commonmark_protectors_complete() -> None:
    body = """# Heading

Paragraph with <strong>inline HTML</strong>.

- parent item
  - nested item
- sibling item

| Key | Value |
| --- | --- |
| A | B |

> quoted paragraph

```python
print("safe")
```

<section><p>Raw HTML block</p></section>
"""
    projection = _confluence_projection(body)

    catalogs = [request.catalog for request in _requests(projection)]

    assert all(catalog.usable for catalog in catalogs)
    assert all(
        error.code.value != "invalid_authority_range"
        for catalog in catalogs
        for error in catalog.errors
    )
    fragment_types = {
        fragment.fragment_type
        for catalog in catalogs
        for fragment in catalog.fragments
    }
    assert {
        "markdown-heading",
        "markdown-inline-html",
        "markdown-list-item",
        "markdown-table",
        "markdown-blockquote",
        "markdown-code-block",
        "html-p",
    }.issubset(fragment_types)


def test_scoped_reprocess_authorizes_every_current_observation_without_a_delta() -> None:
    initial = _jira_projection(3)
    current = replace(
        initial,
        run_id="run-current",
        deltas=(
            replace(
                initial.deltas[0],
                previous_unit_revision_id=initial.source_unit_revisions[0].id,
                axes=frozenset(),
                changed_anchors=(),
                added_observation_ids=(),
            ),
        ),
    )

    assert not current.deltas[0].requires_extraction
    authority = plan_projection_evidence_work(
        current,
        committed_base_snapshot=_committed_snapshot(initial),
        reprocess_all_current_observations=True,
    )

    assert isinstance(authority, ExtractionAuthority)
    assert authority.ranges_by_observation_id == dict.fromkeys(item.id for item in current.observations)
    primary = _primary(_requests(current, base=initial, committed=_committed_snapshot(initial), reprocess=True))
    # Every provider Observation is read; the Unit Title is never Primary.
    assert {fragment.anchor.observation_id for fragment in primary} == {
        item.id for item in current.observations if item.observation_type != UNIT_TITLE_OBSERVATION_TYPE
    }


def test_every_eligible_image_is_primary_in_exactly_one_request_and_ineligible_originals_are_never_read() -> None:
    projection = _confluence_projection_with_images(12)
    binary_ids = {
        item.id for item in projection.observations if item.observation_type == "binary_artifact"
    }
    requests = _requests(projection, images=_artifact_images(projection))
    primary_artifacts = [
        fragment.anchor.observation_id for fragment in _primary(requests) if fragment.kind.value == "artifact"
    ]
    assert sorted(primary_artifacts) == sorted(binary_ids)

    ineligible = _confluence_projection_with_images(1, inference_eligible=False)
    ineligible_id = next(
        item.id for item in ineligible.observations if item.observation_type == "binary_artifact"
    )
    assert all(
        fragment.anchor.observation_id != ineligible_id
        for request in _requests(ineligible)
        for fragment in request.catalog.fragments
    )


def test_legacy_artifact_without_eligibility_metadata_is_read() -> None:
    projection = _confluence_projection_with_images(1, artifact_size=7)
    binary_id = next(
        observation.id
        for observation in projection.observations
        if observation.observation_type == "binary_artifact"
    )
    revisions = []
    for revision in projection.observation_revisions:
        if revision.observation_id != binary_id:
            revisions.append(revision)
            continue
        metadata = dict(revision.metadata)
        artifact = dict(metadata["source_artifact"])
        artifact.pop("inference_eligible")
        artifact.pop("inference_ineligible_reason")
        metadata["source_artifact"] = artifact
        revisions.append(replace(revision, metadata=metadata))
    legacy_projection = replace(projection, observation_revisions=tuple(revisions))

    primary = _primary(_requests(legacy_projection, images=_artifact_images(legacy_projection)))

    assert binary_id in {fragment.anchor.observation_id for fragment in primary}


def test_preceding_observation_is_the_reply_target_else_the_declared_predecessor() -> None:
    projection = project_source_item(
        source_id="src-teams",
        source_type="teams",
        run_id="run-teams-replies",
        item=ContentItem(
            item_id="teams-window-1",
            title="PCC Agent Dev",
            source_url="https://teams.example.test/conversations/conv-1",
            last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
            version="2",
            extra={"conversation_id": "conv-1", "window_id": "window-1"},
        ),
        raw=RawContent(
            item=ContentItem(item_id="teams-window-1", title="PCC Agent Dev", source_url="", last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc)),
            body=json.dumps({"messages": [
                {"id": "msg-1", "content": "Question", "time": "2026-07-15T10:00:00Z"},
                {"id": "msg-2", "content": "Aside", "time": "2026-07-15T10:01:00Z"},
                {"id": "msg-3", "content": "Answer", "time": "2026-07-15T10:02:00Z", "reply_to_id": "msg-1"},
            ]}).encode(),
            content_type="application/json",
        ),
        normalized=NormalizedContent(
            item=ContentItem(item_id="teams-window-1", title="PCC Agent Dev", source_url="", last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc)),
            markdown_body="normalized Teams window",
        ),
    )
    by_key = {item.provider_key: item.id for item in projection.observations}

    assert preceding_observation_id(projection, by_key["msg-3"]) == by_key["msg-1"]
    assert preceding_observation_id(projection, by_key["msg-2"]) == by_key["msg-1"]
    assert preceding_observation_id(projection, by_key["msg-1"]) is None
    assert preceding_observation_id(projection, by_key["$unit_identity"]) is None

    jira = _jira_projection(2)
    core, first, second = (
        item.id for item in jira.observations if item.observation_type != UNIT_TITLE_OBSERVATION_TYPE
    )
    assert preceding_observation_id(jira, first) == core
    assert preceding_observation_id(jira, second) == first


def test_source_projection_attaches_summary_only_to_exact_image_revision() -> None:
    projection = _confluence_projection_with_images(1)
    artifact_observation = next(
        item
        for item in projection.observations
        if item.observation_type == "binary_artifact"
    )

    enriched = with_source_artifact_summaries(
        projection,
        (
            SourceArtifactSummary(
                source_observation_id=artifact_observation.id,
                summary="Architecture diagram showing the bounded request flow.",
            ),
        ),
    )

    revision = next(
        item
        for item in enriched.observation_revisions
        if item.observation_id == artifact_observation.id
    )
    assert revision.id == next(
        item.id
        for item in projection.observation_revisions
        if item.observation_id == artifact_observation.id
    )
    assert revision.metadata["source_artifact"]["summary"] == (
        "Architecture diagram showing the bounded request flow."
    )

    primary_observation = next(
        item
        for item in projection.observations
        if item.observation_type != "binary_artifact"
    )
    with pytest.raises(ValueError, match="non-Artifact"):
        with_source_artifact_summaries(
            projection,
            (
                SourceArtifactSummary(
                    source_observation_id=primary_observation.id,
                    summary="Invalid target.",
                ),
            ),
        )
    summary = SourceArtifactSummary(
        source_observation_id=artifact_observation.id,
        summary="Duplicate target.",
    )
    with pytest.raises(ValueError, match="must be unique"):
        with_source_artifact_summaries(projection, (summary, summary))


