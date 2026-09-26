from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.pipeline.extraction_requests import plan_extraction_requests
from memforge.pipeline.projection_context import (
    CommittedSourceUnitSnapshot,
    ExtractionAuthority,
    plan_projection_evidence_work,
)
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.source_projection_adapters import (
    BUILTIN_SPECIALIZED_SOURCE_TYPES,
    project_source_item,
)
from tests.llm_fixture import NoopMemoryExtractor


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


@pytest.mark.parametrize(
    "source_type",
    [*TEXTUAL_BUILTIN_SOURCE_TYPES, "extension_document"],
)
def test_active_v9_source_matrix_uses_exact_current_incremental_authority(
    source_type: str,
) -> None:
    source_id = f"src-v9-{source_type}"
    first_item, first_raw, first_normalized, target_type = _projection_inputs(
        source_type,
        OLD_RULE,
        version="1",
    )
    first = project_source_item(
        source_id=source_id,
        source_type=source_type,
        run_id="run-v9-1",
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
        run_id="run-v9-2",
        item=second_item,
        raw=second_raw,
        normalized=second_normalized,
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision
            for revision in first.observation_revisions
        },
    )
    target_observation_id = next(
        observation.id
        for observation in second.observations
        if observation.observation_type == target_type
    )

    authority = plan_projection_evidence_work(
        second,
        committed_base_snapshot=CommittedSourceUnitSnapshot(
            unit_revision=first.source_unit_revisions[0],
            observation_revisions=first.observation_revisions,
        ),
        reprocess_all_current_observations=False,
    )

    assert isinstance(authority, ExtractionAuthority)
    assert target_observation_id in authority.ranges_by_observation_id
    requests = plan_extraction_requests(
        RevisionAssessmentContext(projection=second, base=first, access_context_hash="workspace"),
        authority,
        extractor=NoopMemoryExtractor(),
        source_type=source_type,
        doc_type="document",
    ).requests
    primary_text = "\n".join(
        fragment.presentation_text
        for request in requests
        for fragment in request.catalog.fragments
        if fragment.primary_eligible
    )
    assert CURRENT_RULE in primary_text
    assert OLD_RULE not in primary_text


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


