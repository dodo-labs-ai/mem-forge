from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.llm.structured import (
    MemoryCandidate,
    MemoryExtractionResponse,
    StructuredLlmImage,
)
from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.projection_context import plan_projection_extraction_batches
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_projection import AnchorKind, DeltaAxis, SourceAnchor
from memforge.source_artifacts import StoredSourceArtifact


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


def _confluence_projection_with_images(image_count: int):
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
            media_type="image/png",
            size_bytes=10,
            sha256=f"{index:064x}",
            uri=f"source-artifacts/src-c/artifact-{index}.png",
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
def _teams_projection():
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
            {"id": "msg-1", "content": "Decision: keep A7", "time": "2026-07-15T10:00:00Z"},
            {"id": "msg-2", "content": "Acknowledged: keep A7", "time": "2026-07-15T10:01:00Z"},
        ]
    }
    return project_source_item(
        source_id="src-teams",
        source_type="teams",
        run_id="run-teams",
        item=item,
        raw=RawContent(item=item, body=json.dumps(messages).encode(), content_type="application/json"),
        normalized=NormalizedContent(item=item, markdown_body="normalized Teams window"),
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


def test_many_messages_use_bounded_transient_batches_not_persisted_units() -> None:
    projection = _jira_projection(20)

    batches = plan_projection_extraction_batches(
        projection,
        max_primary_observations=8,
    )

    assert len(batches) == 3
    assert all(len(batch.primary_observation_ids) <= 8 for batch in batches)
    assert {batch.source_unit_id for batch in batches} == {projection.source_units[0].id}


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


def test_one_large_document_is_range_sliced_without_creating_finer_source_units() -> None:
    body = "\n".join(f"line-{index:04d}" for index in range(300))
    projection = _confluence_projection(body)

    batches = plan_projection_extraction_batches(
        projection,
        max_primary_chars=500,
        primary_overlap_chars=50,
    )

    assert len(projection.source_units) == 1
    assert len(batches) > 1
    assert all(len(batch.primary_markdown) <= 500 for batch in batches)
    assert {batch.source_unit_id for batch in batches} == {projection.source_units[0].id}
    assert {batch.primary_observation_ids for batch in batches} == {
        (projection.observations[0].id,)
    }
    rendered = "\n".join(batch.primary_markdown for batch in batches)
    assert "line-0000" in rendered
    assert "line-0299" in rendered


def test_default_large_page_batches_bound_primary_output_pressure() -> None:
    body = "durable meeting decision\n" * 6_000
    projection = _confluence_projection(body)

    batches = plan_projection_extraction_batches(projection)

    assert len(body) > 120_000
    assert len(batches) >= 5
    assert all(len(batch.primary_markdown) <= 30_000 for batch in batches)
    assert {batch.source_unit_id for batch in batches} == {projection.source_units[0].id}
    assert {batch.primary_observation_ids for batch in batches} == {
        (projection.observations[0].id,)
    }


@pytest.mark.asyncio
async def test_projection_batch_extractor_rejects_claim_grounded_only_in_context() -> None:
    projection = _jira_projection(3)
    batch = plan_projection_extraction_batches(
        projection,
        max_primary_observations=1,
    )[2]

    class Client:
        async def extract_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            return MemoryExtractionResponse(
                memories=[
                    MemoryCandidate(
                        content="A7 is retained.",
                        memory_type="decision",
                        confidence=0.9,
                        entity_refs=[],
                        extraction_context="Reply 1: retain A7",
                        evidence_quote="Reply 1: retain A7",
                    ),
                    MemoryCandidate(
                        content="Context-only claim.",
                        memory_type="fact",
                        confidence=0.9,
                        entity_refs=[],
                        extraction_context="A7 processing context",
                        evidence_quote="A7 processing context",
                    ),
                ]
            )

    result = await MemoryExtractor(structured_llm_client=Client()).extract_projection_batch_memories(
        batch,
        source_type="jira",
    )

    assert [memory.content for memory in result.memories] == ["A7 is retained."]
    assert result.memories[0].source_observation_id == batch.primary_observation_ids[0]
    assert result.metadata["structured_llm_calls"] == 1
    assert result.metadata["prompt_chars"] > 0


@pytest.mark.asyncio
async def test_projection_batch_extractor_accepts_only_explicit_visual_evidence() -> None:
    projection = _jira_projection(1)
    batch = plan_projection_extraction_batches(projection)[0]
    visual_observation_id = batch.primary_observation_ids[-1]
    observed_images = ()

    class Client:
        async def extract_memories(self, prompt: str, **kwargs):
            nonlocal observed_images
            del prompt
            observed_images = kwargs["images"]
            return MemoryExtractionResponse(
                memories=[
                    MemoryCandidate(
                        content="The screenshot shows a settled validation result.",
                        memory_type="fact",
                        evidence_quote="",
                        source_observation_id=visual_observation_id,
                    ),
                    MemoryCandidate(
                        content="An unbound visual claim must be rejected.",
                        memory_type="fact",
                        evidence_quote="",
                    ),
                ]
            )

    image = StructuredLlmImage(
        source_observation_id=visual_observation_id,
        media_type="image/png",
        body=b"\x89PNG",
    )
    result = await MemoryExtractor(structured_llm_client=Client()).extract_projection_batch_memories(
        batch,
        source_type="jira",
        images=(image,),
    )

    assert observed_images == (image,)
    assert [item.content for item in result.memories] == [
        "The screenshot shows a settled validation result."
    ]
    assert result.memories[0].source_observation_id == visual_observation_id
    assert result.memories[0].evidence_anchor == "source_artifact"
    assert result.memories[0].evidence_quote is None


@pytest.mark.asyncio
async def test_projection_batch_extractor_preserves_declared_required_context() -> None:
    projection = _jira_projection(3)
    batch = plan_projection_extraction_batches(
        projection,
        max_primary_observations=1,
    )[2]
    required_id = batch.context_observation_ids[0]

    class Client:
        async def extract_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            return MemoryExtractionResponse(
                memories=[
                    MemoryCandidate(
                        content="A7 is retained under the issue context.",
                        memory_type="decision",
                        evidence_quote="Reply 1: retain A7",
                        source_observation_id=batch.primary_observation_ids[0],
                        required_source_observation_ids=[required_id],
                    ),
                    MemoryCandidate(
                        content="An invented dependency must be rejected.",
                        memory_type="decision",
                        evidence_quote="Reply 1: retain A7",
                        source_observation_id=batch.primary_observation_ids[0],
                        required_source_observation_ids=["obs-not-in-context"],
                    ),
                ]
            )

    result = await MemoryExtractor(structured_llm_client=Client()).extract_projection_batch_memories(
        batch,
        source_type="jira",
    )

    assert [item.content for item in result.memories] == [
        "A7 is retained under the issue context."
    ]
    assert result.memories[0].required_source_observation_ids == [required_id]


@pytest.mark.asyncio
async def test_projection_batch_rejects_context_that_belongs_only_to_another_primary() -> None:
    projection = _jira_projection(3)
    comments = [item for item in projection.observations if item.observation_type == "comment"]
    revisions = {item.observation_id: item for item in projection.observation_revisions}
    changed = comments[:2]
    delta = replace(
        projection.deltas[0],
        axes=frozenset({DeltaAxis.SEMANTIC}),
        changed_anchors=tuple(
            SourceAnchor(
                kind=AnchorKind.WHOLE_OBSERVATION,
                observation_id=item.id,
                observation_revision_id=revisions[item.id].id,
            )
            for item in changed
        ),
        added_observation_ids=(),
    )
    projection = replace(projection, deltas=(delta,))
    batch = plan_projection_extraction_batches(projection)[0]
    first_primary_id = changed[0].id
    context_for_other_primary_id = comments[2].id
    assert context_for_other_primary_id in batch.context_observation_ids

    class Client:
        async def extract_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            return MemoryExtractionResponse(
                memories=[
                    MemoryCandidate(
                        content="The first comment depends on non-adjacent context.",
                        memory_type="decision",
                        evidence_quote="Reply 0: retain A7",
                        source_observation_id=first_primary_id,
                        required_source_observation_ids=[context_for_other_primary_id],
                    ),
                    MemoryCandidate(
                        content="The second comment depends on its adjacent context.",
                        memory_type="decision",
                        evidence_quote="Reply 1: retain A7",
                        source_observation_id=changed[1].id,
                        required_source_observation_ids=[context_for_other_primary_id],
                    ),
                ]
            )

    result = await MemoryExtractor(structured_llm_client=Client()).extract_projection_batch_memories(
        batch,
        source_type="jira",
    )

    assert [item.content for item in result.memories] == [
        "The second comment depends on its adjacent context."
    ]


@pytest.mark.asyncio
async def test_projection_batch_extractor_uses_explicit_observation_for_duplicate_quote() -> None:
    projection = _jira_projection(2)
    batches = plan_projection_extraction_batches(
        projection,
        max_primary_observations=3,
    )
    comment_batch = batches[0]
    first_id, second_id = comment_batch.primary_observation_ids[-2:]
    duplicate_quote = "retain A7"

    class Client:
        async def extract_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            return MemoryExtractionResponse(
                memories=[
                    MemoryCandidate(
                        content="The second reply retains A7.",
                        memory_type="decision",
                        evidence_quote=duplicate_quote,
                        source_observation_id=second_id,
                    ),
                    MemoryCandidate(
                        content="An unanchored duplicate must be skipped.",
                        memory_type="decision",
                        evidence_quote=duplicate_quote,
                    ),
                    MemoryCandidate(
                        content="A mismatched explicit anchor must be skipped.",
                        memory_type="decision",
                        evidence_quote="Reply 0: retain A7",
                        source_observation_id=second_id,
                    ),
                ]
            )

    result = await MemoryExtractor(structured_llm_client=Client()).extract_projection_batch_memories(
        comment_batch,
        source_type="jira",
    )

    assert [memory.content for memory in result.memories] == ["The second reply retains A7."]
    assert result.memories[0].source_observation_id == second_id
    assert first_id != second_id


@pytest.mark.asyncio
async def test_teams_batch_preserves_message_observation_anchor() -> None:
    projection = _teams_projection()
    batch = plan_projection_extraction_batches(projection)[0]
    target_id = batch.primary_observation_ids[1]

    class Client:
        async def extract_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            return MemoryExtractionResponse(
                memories=[
                    MemoryCandidate(
                        content="A7 remains enabled.",
                        memory_type="decision",
                        evidence_quote="keep A7",
                        source_observation_id=target_id,
                    )
                ]
            )

    result = await MemoryExtractor(structured_llm_client=Client()).extract_projection_batch_memories(
        batch,
        source_type="teams",
    )

    assert len(result.memories) == 1
    assert result.memories[0].source_observation_id == target_id
