from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.llm.structured import MemoryCandidate, MemoryExtractionResponse
from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.projection_context import plan_projection_extraction_batches
from memforge.pipeline.source_projection_adapters import project_source_item


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
                        tags=["payroll"],
                        extraction_context="Reply 1: retain A7",
                        evidence_quote="Reply 1: retain A7",
                    ),
                    MemoryCandidate(
                        content="Context-only claim.",
                        memory_type="fact",
                        confidence=0.9,
                        entity_refs=[],
                        tags=["payroll"],
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
