from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.pipeline.projection_context import (
    CommittedSourceUnitSnapshot,
    plan_projection_evidence_work,
    plan_projection_extraction_batches,
)
from memforge.pipeline.projection_fragments import compile_projection_fragment_catalog
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_projection import (
    AnchorKind,
    SourceAnchor,
    with_source_artifact_summaries,
)
from memforge.source_artifacts import (
    SourceArtifactSummary,
    StoredSourceArtifact,
)


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


def test_jira_short_comments_are_batched_with_core_and_adjacent_context() -> None:
    projection = _jira_projection(3)

    batches = plan_projection_extraction_batches(
        projection,
        max_primary_observations=1,
    )

    comment_batch = batches[2]
    assert len(comment_batch.primary_observation_ids) == 1
    assert "Reply 1: retain A7" in comment_batch.primary_markdown
    assert "A7 processing context" in comment_batch.context_markdown
    assert "Reply 0: retain A7" in comment_batch.context_markdown
    assert "Reply 2: retain A7" in comment_batch.context_markdown
    assert "Reply 1: retain A7" in dict(comment_batch.primary_content_by_observation_id)[
        comment_batch.primary_observation_ids[0]
    ]


def test_all_bounded_text_context_is_candidate_context_independent_of_relation_type() -> None:
    projection = _jira_projection(3)
    comments = [
        observation
        for observation in projection.observations
        if observation.observation_type == "comment"
    ]
    revisions = {
        revision.observation_id: revision
        for revision in projection.observation_revisions
    }
    changed = comments[1]
    projection = replace(
        projection,
        deltas=(
            replace(
                projection.deltas[0],
                changed_anchors=(
                    SourceAnchor(
                        kind=AnchorKind.WHOLE_OBSERVATION,
                        observation_id=changed.id,
                        observation_revision_id=revisions[changed.id].id,
                    ),
                ),
                added_observation_ids=(),
            ),
        ),
    )

    [batch] = plan_projection_extraction_batches(projection)

    assert batch.primary_observation_ids == (changed.id,)
    assert set(batch.candidate_context_observation_ids) == set(
        batch.context_observation_ids
    )


def test_many_messages_use_bounded_transient_batches_not_persisted_units() -> None:
    projection = _jira_projection(20)

    batches = plan_projection_extraction_batches(
        projection,
        max_primary_observations=8,
    )

    assert len(batches) == 3
    assert all(len(batch.primary_observation_ids) <= 8 for batch in batches)
    assert {batch.source_unit_id for batch in batches} == {projection.source_units[0].id}


def test_scoped_replay_can_select_all_current_observations_without_a_delta() -> None:
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

    batches = plan_projection_extraction_batches(
        current,
        primary_observation_ids=tuple(
            observation.id for observation in current.observations
        ),
        max_primary_observations=1,
    )

    assert {
        observation_id
        for batch in batches
        for observation_id in batch.primary_observation_ids
    } == {observation.id for observation in current.observations}


def test_projection_batch_records_authority_segmentation_policy_identity() -> None:
    projection = _jira_projection(1)

    [batch] = plan_projection_extraction_batches(
        projection,
        max_primary_observations=8,
    )

    assert batch.authority_policy_version == 5
    assert batch.id == "xbatch-907c4a188b1d37bf"


def test_many_images_use_bounded_multimodal_batches_without_losing_artifacts() -> None:
    projection = _confluence_projection_with_images(56)

    batches = plan_projection_extraction_batches(projection)
    binary_observation_ids = {
        observation.id
        for observation in projection.observations
        if observation.observation_type == "binary_artifact"
    }
    batched_binary_ids = [
        observation_id
        for batch in batches
        for observation_id in batch.primary_observation_ids
        if observation_id in binary_observation_ids
    ]

    assert len(binary_observation_ids) == 56
    assert set(batched_binary_ids) == binary_observation_ids
    assert len(batched_binary_ids) == len(binary_observation_ids)
    assert all(
        len(binary_observation_ids.intersection(batch.primary_observation_ids)) <= 8
        for batch in batches
    )


def test_multimodal_batches_bound_bytes_and_exclude_ineligible_originals() -> None:
    eligible = _confluence_projection_with_images(3, artifact_size=7)
    batches = plan_projection_extraction_batches(
        eligible,
        max_primary_binary_bytes=10,
    )
    binary_ids = {
        observation.id
        for observation in eligible.observations
        if observation.observation_type == "binary_artifact"
    }
    assert all(
        len(binary_ids.intersection(batch.primary_observation_ids)) <= 1
        for batch in batches
    )
    assert [batch.primary_image_bytes for batch in batches] == [7, 7, 7]

    ineligible = _confluence_projection_with_images(
        1,
        artifact_size=11,
        inference_eligible=False,
    )
    ineligible_binary_id = next(
        observation.id
        for observation in ineligible.observations
        if observation.observation_type == "binary_artifact"
    )
    ineligible_batches = plan_projection_extraction_batches(
        ineligible,
        max_primary_binary_bytes=10,
    )
    assert all(
        ineligible_binary_id not in batch.primary_observation_ids
        for batch in ineligible_batches
    )
    assert all(
        ineligible_binary_id not in batch.context_observation_ids
        for batch in ineligible_batches
    )
    assert any(
        any(
            observation_id != ineligible_binary_id
            for observation_id in batch.primary_observation_ids
        )
        for batch in ineligible_batches
    )


def test_candidate_context_artifacts_share_the_batch_binary_budget() -> None:
    projection = _confluence_projection_with_images(2, artifact_size=4)
    artifact_ids = [
        observation.id
        for observation in projection.observations
        if observation.observation_type == "binary_artifact"
    ]

    batches = plan_projection_extraction_batches(
        projection,
        max_primary_observations=1,
        max_primary_binary_bytes=5,
    )
    first_artifact_batch = next(
        batch
        for batch in batches
        if batch.primary_observation_ids == (artifact_ids[0],)
    )

    assert first_artifact_batch.primary_image_bytes == 4
    assert artifact_ids[1] in first_artifact_batch.context_observation_ids
    assert artifact_ids[1] not in first_artifact_batch.candidate_context_observation_ids
    assert first_artifact_batch.candidate_context_image_bytes == 0


def test_multimodal_batches_admit_legacy_artifact_without_eligibility_metadata() -> None:
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
    legacy_projection = replace(
        projection,
        observation_revisions=tuple(revisions),
    )

    batches = plan_projection_extraction_batches(
        legacy_projection,
        max_primary_binary_bytes=10,
    )

    assert any(
        binary_id in batch.primary_observation_ids
        for batch in batches
    )


def test_pdf_artifact_does_not_claim_multimodal_image_input_bytes() -> None:
    projection = _confluence_projection_with_images(
        1,
        artifact_size=7,
        media_type="application/pdf",
    )

    batches = plan_projection_extraction_batches(projection)

    assert any(
        observation_id
        for batch in batches
        for observation_id in batch.primary_observation_ids
        if observation_id
        in {
            observation.id
            for observation in projection.observations
            if observation.observation_type == "binary_artifact"
        }
    )
    assert all(batch.primary_image_bytes == 0 for batch in batches)


def test_one_large_document_is_split_into_batches_without_creating_finer_source_units() -> None:
    body = "\n".join(f"line-{index:04d}" for index in range(300))
    projection = _confluence_projection(body)

    batches = plan_projection_extraction_batches(
        projection,
        max_primary_chars=500,
    )

    assert len(projection.source_units) == 1
    assert len(batches) > 1
    assert {batch.source_unit_id for batch in batches} == {projection.source_units[0].id}
    assert {batch.primary_observation_ids for batch in batches} == {
        (projection.observations[0].id,)
    }
    rendered = "\n".join(batch.primary_markdown for batch in batches)
    assert "line-0000" in rendered
    assert "line-0299" in rendered
    revision = projection.observation_revisions[0]
    for batch in batches:
        for observation_id, start, text in batch.primary_authority_spans:
            assert observation_id == projection.observations[0].id
            assert revision.content[start : start + len(text)] == text


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
    batches = plan_projection_evidence_work(
        target,
        committed_base_snapshot=_committed_snapshot(initial),
        reprocess_all_current_observations=False,
    )
    catalogs = [
        compile_projection_fragment_catalog(
            target,
            batch,
            access_context_hash="incremental-authority",
        )
        for batch in batches
    ]
    primary_text = "\n".join(
        fragment.presentation_text
        for catalog in catalogs
        for fragment in catalog.fragments
        if fragment.primary_eligible
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

    batches = plan_projection_evidence_work(
        target,
        committed_base_snapshot=_committed_snapshot(initial),
        reprocess_all_current_observations=False,
    )
    catalogs = [
        compile_projection_fragment_catalog(
            target,
            batch,
            access_context_hash="canonical-field-authority",
        )
        for batch in batches
    ]
    primary_text = [
        fragment.presentation_text
        for catalog in catalogs
        for fragment in catalog.fragments
        if fragment.primary_eligible
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

    batches = plan_projection_evidence_work(
        target,
        committed_base_snapshot=_committed_snapshot(initial),
        reprocess_all_current_observations=False,
    )
    catalogs = [
        compile_projection_fragment_catalog(
            target,
            batch,
            access_context_hash="nested-canonical-authority",
        )
        for batch in batches
    ]
    primary_text = "\n".join(
        fragment.presentation_text
        for catalog in catalogs
        for fragment in catalog.fragments
        if fragment.primary_eligible
    )

    assert changed.strip() in primary_text
    assert "Historical context remains." not in primary_text
    primary = [f for catalog in catalogs for f in catalog.fragments if f.primary_eligible]
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

    batches = plan_projection_evidence_work(
        projection,
        reprocess_all_current_observations=reprocess,
    )

    assert batches == ()


def test_v9_batches_keep_one_crossing_markdown_structure_complete() -> None:
    prefix = ("Short paragraph.\n\n" * 3_100)
    code_block = "```text\n" + ("x" * 12_000) + "\n```\n"
    body = prefix + code_block
    projection = _confluence_projection(body)

    batches = plan_projection_extraction_batches(
        projection,
    )
    catalogs = [
        compile_projection_fragment_catalog(
            projection,
            batch,
            access_context_hash="access-v9-structural",
        )
        for batch in batches
    ]

    assert all(catalog.usable for catalog in catalogs)
    revision_content = projection.observation_revisions[0].content
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


def test_v9_keeps_one_structural_unit_larger_than_the_grouping_target() -> None:
    projection = _confluence_projection(
        "```text\n" + ("x" * 1_000) + "\n```\n"
    )

    batches = plan_projection_extraction_batches(
        projection,
        max_primary_chars=1,
    )
    catalogs = [compile_projection_fragment_catalog(projection, batch, access_context_hash="scope")
                for batch in batches]
    assert all(catalog.usable for catalog in catalogs)
    [fragment] = [f for catalog in catalogs for f in catalog.fragments
                  if f.primary_eligible and f.fragment_type == "markdown-code-block"]
    assert fragment.presentation_text == "```text\n" + ("x" * 1_000) + "\n```"


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

    result = plan_projection_evidence_work(
        target,
        committed_base_snapshot=_committed_snapshot(initial),
        reprocess_all_current_observations=False,
    )

    assert isinstance(result, tuple) and len(result) == 1
    catalog = compile_projection_fragment_catalog(target, result[0], access_context_hash="scope")
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

    batches = plan_projection_extraction_batches(
        projection,
        max_primary_chars=140,
    )
    catalogs = [
        compile_projection_fragment_catalog(
            projection,
            batch,
            access_context_hash="access-v9-protectors",
        )
        for batch in batches
    ]

    assert len(batches) > 1
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


def test_structural_planner_avoids_inline_allocation_but_compiler_still_validates_html(monkeypatch) -> None:
    from markdown_it.parser_inline import ParserInline

    original_parse = ParserInline.parse
    inline_calls = []

    def track_inline(self, src, md, env, tokens):
        inline_calls.append(src)
        return original_parse(self, src, md, env, tokens)

    monkeypatch.setattr(ParserInline, "parse", track_inline)
    projection = _confluence_projection(
        "# Guide\n\n" + "Paragraph with **bold** and <strong>inline HTML</strong>.\n\n" * 8
    )
    batches = plan_projection_extraction_batches(
        projection, max_primary_chars=200,
    )

    assert len(batches) > 1
    assert inline_calls == []

    catalogs = [
        compile_projection_fragment_catalog(projection, batch, access_context_hash="allocation-test")
        for batch in batches
    ]
    assert inline_calls
    assert all(catalog.usable for catalog in catalogs)
    html_fragments = [
        fragment for catalog in catalogs for fragment in catalog.fragments
        if fragment.fragment_type == "markdown-inline-html"
    ]
    assert len(html_fragments) == 8
    assert all("<strong>" not in fragment.presentation_text for fragment in html_fragments)


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


