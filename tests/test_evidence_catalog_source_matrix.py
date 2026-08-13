from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest

from memforge.llm.structured import MemoryCandidate, MemoryExtractionResponse
from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.projection_context import plan_projection_extraction_batches
from memforge.pipeline.projection_evidence import build_projected_claim_evidence
from memforge.pipeline.source_projection_adapters import (
    BUILTIN_SPECIALIZED_SOURCE_TYPES,
    project_source_item,
)


NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)
OLD_RULE = "Use contextId for the first diagnostic step."
CURRENT_RULE = "Use traceId to query the current application logs."
TEXTUAL_BUILTIN_SOURCE_TYPES = (
    "confluence",
    "jira",
    "github_repo",
    "github_pages",
    "local_markdown",
    "teams",
    "agent_session",
)


def test_source_matrix_covers_every_builtin_textual_source_type() -> None:
    assert set(TEXTUAL_BUILTIN_SOURCE_TYPES) == set(BUILTIN_SPECIALIZED_SOURCE_TYPES)


def _projection_inputs(
    source_type: str,
    rule: str,
    *,
    version: str,
) -> tuple[ContentItem, RawContent, NormalizedContent, str]:
    extra: dict[str, object] = {}
    item_id = "doc-1"
    title = "Diagnostic guide"
    source_url = "https://example.test/doc-1"
    raw_value: object = rule
    content_type = "text/plain"
    markdown = rule
    target_observation_type = "document_content"

    if source_type == "confluence":
        item_id = "confluence-42"
        source_url = "https://confluence.example.test/pages/42"
        extra = {"page_id": "42", "space_key": "ENG"}
        target_observation_type = "page_body"
    elif source_type == "jira":
        item_id = "jira-PAY-12"
        source_url = "https://jira.example.test/browse/PAY-12"
        extra = {"issue_id": "10012", "issue_key": "PAY-12"}
        raw_value = {
            "id": "10012",
            "key": "PAY-12",
            "fields": {
                "summary": "Payroll",
                "description": None,
                "status": None,
                "priority": None,
                "assignee": None,
                "labels": [],
                "resolution": None,
                "updated": "2026-08-12T10:00:00Z",
            },
            "_comments": [{"id": "501", "body": rule}],
            "_comments_included": True,
            "_comments_total": 1,
            "changelog": {"startAt": 0, "histories": [], "total": 0},
        }
        content_type = "application/json"
        markdown = f"# PAY-12\n\n{rule}"
        target_observation_type = "comment"
    elif source_type == "github_repo":
        item_id = "github-repo-guide"
        source_url = "https://github.example.test/acme/pay/blob/main/docs/guide.md"
        extra = {
            "repo_owner": "acme",
            "repo_name": "pay",
            "repo_ref": "main",
            "relative_path": "docs/guide.md",
            "file_lineage_id": "file-77",
        }
        target_observation_type = "file_content"
    elif source_type == "github_pages":
        item_id = "github-page-guide"
        source_url = "https://docs.example.test/guide/"
        extra = {"canonical_url": source_url}
        target_observation_type = "page_content"
    elif source_type == "local_markdown":
        item_id = "local-guide"
        source_url = "file:///vault/docs/guide.md"
        extra = {"relative_path": "docs/guide.md", "file_lineage_id": "file-77"}
        raw_value = {
            "vault_id": "vault-a",
            "relative_path": "docs/guide.md",
            "file_lineage_id": "file-77",
            "markdown": rule,
        }
        content_type = "application/json"
        target_observation_type = "file_content"
    elif source_type == "teams":
        item_id = "teams-window-1"
        source_url = "https://teams.example.test/conversations/conversation-1"
        extra = {
            "conversation_id": "conversation-1",
            "window_id": "window-1",
            "root_message_id": "msg-1",
        }
        raw_value = {
            "conversation_id": "conversation-1",
            "window_id": "window-1",
            "messages": [
                {
                    "id": "msg-1",
                    "content": rule,
                    "time": "2026-08-12T10:00:00Z",
                }
            ],
        }
        content_type = "application/json"
        markdown = "normalized Teams window"
        target_observation_type = "message"
    elif source_type == "agent_session":
        item_id = "agent-session-window-1"
        source_url = "memforge://agent-session/session-1"
        raw_value = {
            "doc_id": "agent-session-window-1",
            "markdown": rule,
            "receipt": {
                "client": "codex",
                "session_id": "session-1",
                "history_window_kind": "summary",
            },
        }
        content_type = "application/json"
        target_observation_type = "session_summary"
    elif source_type != "extension_document":
        raise AssertionError(f"unsupported source fixture: {source_type}")

    item = ContentItem(
        item_id=item_id,
        title=title,
        source_url=source_url,
        last_modified=NOW,
        version=version,
        extra=extra,
    )
    body = (
        json.dumps(raw_value).encode("utf-8")
        if content_type == "application/json"
        else str(raw_value).encode("utf-8")
    )
    return (
        item,
        RawContent(item=item, body=body, content_type=content_type),
        NormalizedContent(item=item, markdown_body=markdown),
        target_observation_type,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_type",
    [*TEXTUAL_BUILTIN_SOURCE_TYPES, "extension_document"],
)
async def test_textual_source_matrix_binds_blocks_to_current_stable_observations(
    source_type: str,
) -> None:
    source_id = f"src-{source_type}"
    first_item, first_raw, first_normalized, target_type = _projection_inputs(
        source_type,
        OLD_RULE,
        version="1",
    )
    first = project_source_item(
        source_id=source_id,
        source_type=source_type,
        run_id="run-1",
        item=first_item,
        raw=first_raw,
        normalized=first_normalized,
    )
    second_item, second_raw, second_normalized, _ = _projection_inputs(
        source_type,
        CURRENT_RULE,
        version="2",
    )
    second = project_source_item(
        source_id=source_id,
        source_type=source_type,
        run_id="run-2",
        item=second_item,
        raw=second_raw,
        normalized=second_normalized,
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision
            for revision in first.observation_revisions
        },
    )

    [first_observation] = [
        observation
        for observation in first.observations
        if observation.observation_type == target_type
    ]
    [second_observation] = [
        observation
        for observation in second.observations
        if observation.observation_type == target_type
    ]
    first_revision = next(
        revision
        for revision in first.observation_revisions
        if revision.observation_id == first_observation.id
    )
    second_revision = next(
        revision
        for revision in second.observation_revisions
        if revision.observation_id == second_observation.id
    )

    assert second_observation.id == first_observation.id
    assert second_revision.id != first_revision.id
    assert CURRENT_RULE in second_revision.content

    batches = plan_projection_extraction_batches(second)
    batch = next(
        batch
        for batch in batches
        if second_observation.id in batch.primary_observation_ids
    )
    class Client:
        async def extract_projection_memories(self, prompt: str, **kwargs):
            del kwargs
            rendered_blocks = re.findall(
                r'<evidence_block id="([^"]+)"[^>]*>\n(.*?)\n</evidence_block>',
                prompt,
                re.DOTALL,
            )
            block_id, _ = next(
                (block_id, text)
                for block_id, text in rendered_blocks
                if CURRENT_RULE in text
            )
            return MemoryExtractionResponse(
                memories=[
                    MemoryCandidate(
                        content="The exact current diagnostic procedure is durable.",
                        memory_type="procedure",
                        evidence_block_id=block_id,
                        evidence_quote=CURRENT_RULE,
                    ),
                    MemoryCandidate(
                        content="The fallback current diagnostic procedure is durable.",
                        memory_type="procedure",
                        evidence_block_id=block_id,
                        evidence_quote=(
                            "A provider-formatted paraphrase that is not source text."
                        ),
                    ),
                ]
            )

    result = await MemoryExtractor(
        structured_llm_client=Client()
    ).extract_projection_batch_memories(
        batch,
        source_type=source_type,
    )

    assert len(result.memories) == 2
    exact, fallback = result.memories
    assert exact.evidence_quote == CURRENT_RULE
    assert fallback.evidence_quote
    assert CURRENT_RULE in fallback.evidence_quote
    for memory in result.memories:
        assert memory.evidence_block_id is None
        assert memory.source_observation_id == second_observation.id
        assert (
            second_revision.content[
                memory.evidence_range_start : memory.evidence_range_end
            ]
            == memory.evidence_quote
        )

    staged = build_projected_claim_evidence(
        projection=second,
        raw_memories=result.memories,
        doc_id=second_item.item_id,
        source_type=source_type,
        project_key=None,
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash="workspace",
        extractor_run_id="run-2",
    )
    primary_references = [
        reference
        for reference in staged.references
        if reference.role.value == "primary"
    ]

    assert {unit.excerpt for unit in staged.units} == {
        exact.evidence_quote,
        fallback.evidence_quote,
    }
    assert len(primary_references) == 2
    assert all(
        primary.anchor.observation_id == second_observation.id
        and primary.anchor.observation_revision_id == second_revision.id
        and primary.anchor.observation_revision_id != first_revision.id
        and primary.anchor.kind.value == "revision_range"
        for primary in primary_references
    )
