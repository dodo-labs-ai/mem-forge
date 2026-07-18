from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.pipeline.source_projection_adapters import (
    BUILTIN_SPECIALIZED_SOURCE_TYPES,
    GeneSourceProjectionAdapter,
    project_source_item,
    project_source_unit_tombstone,
    source_run_projection_coverage,
)
from memforge.genes import GENE_REGISTRY
from memforge.source_projection import (
    DeltaAxis,
    ProjectionCoverage,
    ProjectionScopeTransition,
    SourceRelationType,
    SourceUnit,
)
from memforge.source_projection_config import projection_scope_fingerprint
from memforge.storage.database import Database


NOW = datetime(2026, 7, 15, tzinfo=timezone.utc)


def _item(**overrides) -> ContentItem:
    values = dict(
        item_id="doc-1",
        title="Title",
        source_url="https://example.test/doc-1",
        last_modified=NOW,
        version="2",
        extra={},
    )
    values.update(overrides)
    return ContentItem(**values)


def _inputs(item: ContentItem, raw_value: object, markdown: str = "semantic body"):
    raw = RawContent(
        item=item,
        body=(raw_value if isinstance(raw_value, bytes) else json.dumps(raw_value).encode()),
        content_type="text/plain" if isinstance(raw_value, bytes) else "application/json",
    )
    normalized = NormalizedContent(item=item, markdown_body=markdown)
    return raw, normalized


def _jira_payload(
    *,
    field_overrides: dict | None = None,
    comments: list[dict] | None = None,
    comments_total: int | None = None,
    histories: list[dict] | None = None,
    changelog_total: int | None = None,
) -> dict:
    comments = list(comments or [])
    histories = list(histories or [])
    payload = {
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
            "updated": "2026-07-14T10:00:00Z",
            **(field_overrides or {}),
        },
        "_comments": comments,
        "_comments_included": True,
        "_comments_total": len(comments) if comments_total is None else comments_total,
        "changelog": {
            "startAt": 0,
            "histories": histories,
            "total": len(histories) if changelog_total is None else changelog_total,
        },
    }
    if payload["_comments_total"] > len(comments):
        payload["_comments_truncated"] = {
            "returned": len(comments),
            "total": payload["_comments_total"],
        }
    return payload


def _teams_run_attestations(
    transition: ProjectionScopeTransition,
) -> tuple[dict[str, object], ...]:
    conversations = sorted(str(value) for value in transition.target_scope.get("conversation_ids", []))
    return tuple(
        {
            "conversation_id": conversation_id,
            "transition_id": transition.id,
            "target_scope_fingerprint": projection_scope_fingerprint(transition.target_scope),
            "target_conversation_ids": conversations,
            "collection_attempt_id": "job-a:attempt:1",
            "poll": {
                "raw_conversation_id": conversation_id,
                "access_probe_status": "ok",
                "pagination_complete": True,
                "stop_reason": "no_backward_link",
            },
        }
        for conversation_id in conversations
    )


def test_confluence_page_id_is_unit_and_parent_is_location_only() -> None:
    item = _item(
        item_id="confluence-123",
        extra={"page_id": "123", "space_key": "ENG", "parent_page_id": "100"},
    )
    raw, normalized = _inputs(item, b"<p>Payroll result is retained.</p>")
    first = project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-1",
        item=item,
        raw=raw,
        normalized=normalized,
    )
    moved_item = _item(
        item_id="confluence-123",
        source_url="https://example.test/new-parent/doc-1",
        extra={"page_id": "123", "space_key": "ENG", "parent_page_id": "200"},
    )
    moved_raw, moved_normalized = _inputs(moved_item, b"<p>Payroll result is retained.</p>")
    moved = project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-2",
        item=moved_item,
        raw=moved_raw,
        normalized=moved_normalized,
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions={
            first.observations[0].id: first.observation_revisions[0],
        },
    )

    assert first.source_units[0].provider_key == "123"
    assert moved.deltas[0].axes == frozenset({DeltaAxis.LOCATION})
    assert moved.deltas[0].requires_extraction is False
    assert moved.relations[0].relation_type is SourceRelationType.CONTAINED_BY


def test_confluence_title_change_is_semantic_not_location() -> None:
    item = _item(
        item_id="confluence-123",
        title="Old title",
        extra={"page_id": "123", "space_key": "ENG"},
    )
    raw, normalized = _inputs(item, b"<p>Same body.</p>", markdown="Same body.")
    first = project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-title-1",
        item=item,
        raw=raw,
        normalized=normalized,
    )
    renamed = _item(
        item_id="confluence-123",
        title="New title",
        extra={"page_id": "123", "space_key": "ENG"},
    )
    renamed_raw, renamed_normalized = _inputs(
        renamed,
        b"<p>Same body.</p>",
        markdown="Same body.",
    )

    changed = project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-title-2",
        item=renamed,
        raw=renamed_raw,
        normalized=renamed_normalized,
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions={
            first.observations[0].id: first.observation_revisions[0],
        },
    )

    assert changed.deltas[0].axes == frozenset({DeltaAxis.SEMANTIC})
    assert changed.deltas[0].requires_extraction is True


def test_confluence_operational_display_header_does_not_trigger_extraction() -> None:
    item = _item(item_id="confluence-123", extra={"page_id": "123", "space_key": "ENG"})
    raw = RawContent(item=item, body=b"<p>Keep A7.</p>", content_type="text/html")
    first_normalized = NormalizedContent(
        item=item,
        markdown_body="# Title\n**Last modified**: 2026-07-14\n\nKeep A7.",
        source_semantics={"semantic_markdown": "Keep A7."},
    )
    first = project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-operational-1",
        item=item,
        raw=raw,
        normalized=first_normalized,
    )
    later_normalized = NormalizedContent(
        item=item,
        markdown_body="# Title\n**Last modified**: 2026-07-15\n\nKeep A7.",
        source_semantics={"semantic_markdown": "Keep A7."},
    )
    later = project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-operational-2",
        item=item,
        raw=raw,
        normalized=later_normalized,
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions={first.observations[0].id: first.observation_revisions[0]},
    )

    assert later.observation_revisions[0].content == "# Title\n\nKeep A7."
    assert later.deltas[0].axes == frozenset()
    assert later.deltas[0].requires_extraction is False


def test_access_change_revises_unit_without_semantic_extraction() -> None:
    item = _item(item_id="confluence-123", extra={"page_id": "123", "space_key": "ENG"})
    raw, normalized = _inputs(item, b"<p>Keep A7.</p>", markdown="Keep A7.")
    first = project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-access-1",
        item=item,
        raw=raw,
        normalized=normalized,
        access_context={"access_policy": "private", "owner_user_id": "user-a"},
    )
    changed = project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-access-2",
        item=item,
        raw=raw,
        normalized=normalized,
        access_context={"access_policy": "workspace", "owner_user_id": "user-a"},
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions={first.observations[0].id: first.observation_revisions[0]},
    )

    assert changed.source_unit_revisions[0].id != first.source_unit_revisions[0].id
    assert changed.deltas[0].axes == frozenset({DeltaAxis.ACCESS})
    assert changed.deltas[0].requires_extraction is False


def test_jira_numeric_issue_id_is_unit_and_comments_are_observations() -> None:
    item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
    raw, normalized = _inputs(
        item,
        _jira_payload(
            field_overrides={"description": "Context", "status": {"name": "Done"}},
            comments=[
                {"id": "501", "body": "First short reply", "created": "2026-07-14T10:00:00Z"},
                {"id": "502", "body": "Correction: retain A7", "created": "2026-07-14T10:01:00Z"},
            ],
        ),
    )

    projection = project_source_item(
        source_id="src-j",
        source_type="jira",
        run_id="run-j",
        item=item,
        raw=raw,
        normalized=normalized,
    )

    assert projection.source_units[0].provider_key == "10012"
    assert [item.observation_type for item in projection.observations] == [
        "issue_core",
        "comment",
        "comment",
    ]
    assert all(item.source_unit_id == projection.source_units[0].id for item in projection.observations)


def test_jira_projection_rejects_mutable_issue_key_as_unit_identity() -> None:
    item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
    raw, normalized = _inputs(
        item,
        {**_jira_payload(), "id": None},
    )

    with pytest.raises(ValueError, match="stable provider id"):
        project_source_item(
            source_id="src-j",
            source_type="jira",
            run_id="run-j-missing-id",
            item=item,
            raw=raw,
            normalized=normalized,
        )


def test_jira_projection_rejects_comment_without_stable_provider_id() -> None:
    item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
    raw, normalized = _inputs(
        item,
        _jira_payload(comments=[{"body": "Decision without provider identity"}]),
    )

    with pytest.raises(ValueError, match="stable provider id"):
        project_source_item(
            source_id="src-j",
            source_type="jira",
            run_id="run-j-missing-comment-id",
            item=item,
            raw=raw,
            normalized=normalized,
        )


def test_truncated_jira_comments_force_partial_coverage() -> None:
    item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
    raw, normalized = _inputs(
        item,
        _jira_payload(comments_total=10),
    )
    projection = project_source_item(
        source_id="src-j",
        source_type="jira",
        run_id="run-j",
        item=item,
        raw=raw,
        normalized=normalized,
    )

    assert projection.coverage is ProjectionCoverage.PARTIAL_PROJECTION


def test_partial_jira_projection_carries_unreturned_prior_observations() -> None:
    item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
    first_raw, first_normalized = _inputs(
        item,
        _jira_payload(comments=[{"id": "501", "body": "Keep A7"}]),
    )
    first = project_source_item(
        source_id="src-j",
        source_type="jira",
        run_id="run-j-full",
        item=item,
        raw=first_raw,
        normalized=first_normalized,
    )
    partial_raw, partial_normalized = _inputs(
        item,
        _jira_payload(field_overrides={"summary": "Payroll updated"}, comments_total=1),
    )
    partial = project_source_item(
        source_id="src-j",
        source_type="jira",
        run_id="run-j-partial",
        item=item,
        raw=partial_raw,
        normalized=partial_normalized,
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions={revision.observation_id: revision for revision in first.observation_revisions},
    )

    prior_comment = next(
        revision for revision in first.observation_revisions if revision.observation_id != first.observations[0].id
    )
    carried = next(
        revision
        for revision in partial.observation_revisions
        if revision.observation_id == prior_comment.observation_id
    )
    assert partial.coverage is ProjectionCoverage.PARTIAL_PROJECTION
    assert carried.id == prior_comment.id
    assert carried == prior_comment
    assert partial.carried_observation_revision_ids == (prior_comment.id,)
    assert partial.deltas[0].removed_observation_ids == ()


@pytest.mark.asyncio
async def test_partial_jira_projection_reuses_immutable_carried_revision_in_store(tmp_path) -> None:
    database = Database(str(tmp_path / "partial-jira.db"))
    await database.connect()
    await database.upsert_source(
        id="src-j",
        type="jira",
        name="Jira",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    try:
        item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
        first_raw, first_normalized = _inputs(
            item,
            _jira_payload(comments=[{"id": "501", "body": "Keep A7"}]),
        )
        first = project_source_item(
            source_id="src-j",
            source_type="jira",
            run_id="run-j-full-store",
            item=item,
            raw=first_raw,
            normalized=first_normalized,
        )
        await database.record_source_projection(first)

        partial_raw, partial_normalized = _inputs(
            item,
            _jira_payload(field_overrides={"summary": "Payroll updated"}, comments_total=1),
        )
        partial = project_source_item(
            source_id="src-j",
            source_type="jira",
            run_id="run-j-partial-store",
            item=item,
            raw=partial_raw,
            normalized=partial_normalized,
            prior_unit_revision=first.source_unit_revisions[0],
            prior_observation_revisions={revision.observation_id: revision for revision in first.observation_revisions},
        )

        await database.record_source_projection(partial)

        prior_comment = next(
            revision for revision in first.observation_revisions if revision.observation_id != first.observations[0].id
        )
        stored = await database.get_source_projection(partial.run_id)
        assert stored is not None
        carried = next(
            revision
            for revision in stored.observation_revisions
            if revision.observation_id == prior_comment.observation_id
        )
        assert carried == prior_comment
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_unchanged_jira_changelog_reuses_exact_prior_revision_in_store(tmp_path) -> None:
    database = Database(str(tmp_path / "unchanged-jira.db"))
    await database.connect()
    await database.upsert_source(
        id="src-j",
        type="jira",
        name="Jira",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    try:
        item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
        raw, normalized = _inputs(
            item,
            _jira_payload(
                histories=[
                    {
                        "id": "attachment",
                        "items": [{"field": "Attachment"}],
                    }
                ]
            ),
        )
        first = project_source_item(
            source_id="src-j",
            source_type="jira",
            run_id="run-j-unchanged-first",
            item=item,
            raw=raw,
            normalized=normalized,
        )
        original_changelog = next(
            revision
            for revision in first.observation_revisions
            if revision.metadata.get("semantic_class") == "attachment_event"
        )
        stored_changelog = replace(
            original_changelog,
            metadata={"provider_key": "attachment"},
        )
        stored_first = replace(
            first,
            observation_revisions=tuple(
                stored_changelog if revision.id == original_changelog.id else revision
                for revision in first.observation_revisions
            ),
        )
        await database.record_source_projection(stored_first)

        second = project_source_item(
            source_id="src-j",
            source_type="jira",
            run_id="run-j-unchanged-second",
            item=item,
            raw=raw,
            normalized=normalized,
            prior_unit_revision=stored_first.source_unit_revisions[0],
            prior_observation_revisions={
                revision.observation_id: revision
                for revision in stored_first.observation_revisions
            },
        )

        await database.record_source_projection(second)

        projected_changelog = next(
            revision
            for revision in second.observation_revisions
            if revision.observation_id == stored_changelog.observation_id
        )
        assert projected_changelog == stored_changelog
    finally:
        await database.close()


def test_incomplete_embedded_jira_changelog_forces_partial_coverage() -> None:
    item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
    raw, normalized = _inputs(
        item,
        _jira_payload(histories=[{"id": "1"}], changelog_total=2),
    )

    projection = project_source_item(
        source_id="src-j",
        source_type="jira",
        run_id="run-j-changelog-partial",
        item=item,
        raw=raw,
        normalized=normalized,
    )

    assert projection.coverage is ProjectionCoverage.PARTIAL_PROJECTION


def test_incomplete_local_agent_jira_changelog_forces_partial_coverage() -> None:
    item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
    raw, normalized = _inputs(
        item,
        {
            "package_kind": "jira_document",
            "raw_payload": _jira_payload(histories=[{"id": "1"}], changelog_total=2),
        },
    )

    projection = project_source_item(
        source_id="src-j",
        source_type="jira",
        run_id="run-j-local-changelog-partial",
        item=item,
        raw=raw,
        normalized=normalized,
    )

    assert projection.coverage is ProjectionCoverage.PARTIAL_PROJECTION


def test_jira_projection_types_changelog_semantics_for_generic_quality_policy() -> None:
    item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
    raw, normalized = _inputs(
        item,
        _jira_payload(
            histories=[
                {"id": "attachment", "items": [{"field": "Attachment"}]},
                {"id": "routing", "items": [{"field": "priority"}]},
                {"id": "domain", "items": [{"field": "description"}]},
            ]
        ),
    )

    projection = project_source_item(
        source_id="src-j",
        source_type="jira",
        run_id="run-j-typed-changelog",
        item=item,
        raw=raw,
        normalized=normalized,
    )

    semantics = {
        str(revision.metadata["provider_key"]): revision.metadata.get(
            "semantic_class"
        )
        for revision in projection.observation_revisions
        if str(revision.metadata["provider_key"]) in {"attachment", "routing", "domain"}
    }
    assert semantics == {
        "attachment": "attachment_event",
        "routing": "operational_transition",
        "domain": "domain_transition",
    }


@pytest.mark.parametrize("source_type", ["jira", "teams"])
def test_operational_message_metadata_does_not_create_semantic_revision(source_type: str) -> None:
    if source_type == "jira":
        item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
        first_payload = _jira_payload(
            comments=[{"id": "501", "body": "Keep A7", "updated": "2026-07-14T10:00:00Z"}]
        )
        second_payload = {
            **first_payload,
            "_comments": [
                {
                    "id": "501",
                    "body": "Keep A7",
                    "updated": "2026-07-15T10:00:00Z",
                    "author": {"displayName": "Renamed User"},
                }
            ],
        }
    else:
        item = _item(extra={"conversation_id": "conv-1", "window_id": "window-1"})
        first_payload = {
            "messages": [{"id": "msg-1", "content": "Keep A7", "time": "2026-07-14T10:00:00Z"}]
        }
        second_payload = {
            "messages": [
                {
                    "id": "msg-1",
                    "content": "Keep A7",
                    "time": "2026-07-14T10:00:00Z",
                    "lastModifiedDateTime": "2026-07-15T10:00:00Z",
                    "from": {"displayName": "Renamed User"},
                }
            ]
        }
    first_raw, first_normalized = _inputs(item, first_payload)
    first = project_source_item(
        source_id=f"src-{source_type}",
        source_type=source_type,
        run_id="run-operational-1",
        item=item,
        raw=first_raw,
        normalized=first_normalized,
    )
    second_raw, second_normalized = _inputs(item, second_payload)
    second = project_source_item(
        source_id=f"src-{source_type}",
        source_type=source_type,
        run_id="run-operational-2",
        item=item,
        raw=second_raw,
        normalized=second_normalized,
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions={
            observation.id: revision
            for observation, revision in zip(
                first.observations,
                first.observation_revisions,
                strict=True,
            )
        },
    )

    assert second.source_unit_revisions[0].id == first.source_unit_revisions[0].id
    assert second.observation_revisions == first.observation_revisions
    assert second.deltas[0].axes == frozenset()
    assert second.deltas[0].requires_extraction is False


@pytest.mark.parametrize(
    ("source_type", "extra", "raw_value", "expected_unit_type", "expected_observation_type"),
    [
        (
            "github_repo",
            {"repo_owner": "acme", "repo_name": "pay", "relative_path": "docs/design.md", "repo_ref": "main"},
            b"# Design\nKeep A7.",
            "github_file",
            "file_content",
        ),
        (
            "github_pages",
            {"canonical_url": "https://docs.example.test/design/"},
            b"<article>Keep A7.</article>",
            "rendered_page",
            "page_content",
        ),
        (
            "local_markdown",
            {"relative_path": "design.md"},
            {
                "vault_id": "vault-a",
                "relative_path": "design.md",
                "file_lineage_id": "file-77",
                "markdown": "Keep A7.",
            },
            "local_file",
            "file_content",
        ),
        (
            "agent_session",
            {},
            {
                "receipt": {"client": "codex", "session_id": "session-1", "history_window_kind": "summary"},
                "markdown": "The user confirmed A7.",
            },
            "agent_session_window",
            "session_summary",
        ),
    ],
)
def test_document_and_append_sources_use_stable_provider_units(
    source_type: str,
    extra: dict,
    raw_value: object,
    expected_unit_type: str,
    expected_observation_type: str,
) -> None:
    item = _item(extra=extra)
    raw, normalized = _inputs(item, raw_value)
    projection = project_source_item(
        source_id=f"src-{source_type}",
        source_type=source_type,
        run_id=f"run-{source_type}",
        item=item,
        raw=raw,
        normalized=normalized,
    )

    assert projection.source_units[0].unit_type == expected_unit_type
    assert projection.observations[0].observation_type == expected_observation_type


@pytest.mark.parametrize("source_type", ["github_repo", "local_markdown"])
def test_file_move_with_provider_lineage_preserves_observation_identity(source_type: str) -> None:
    first_extra = {"relative_path": "old/design.md", "file_lineage_id": "file-77"}
    second_extra = {
        "relative_path": "new/design.md",
        "file_lineage_id": "file-77",
        "previous_filename": "old/design.md",
    }
    if source_type == "github_repo":
        first_extra.update({"repo_owner": "acme", "repo_name": "pay", "repo_ref": "main"})
        second_extra.update({"repo_owner": "acme", "repo_name": "pay", "repo_ref": "main"})
    first_item = _item(item_id="file-old", extra=first_extra)
    first_raw_value = (
        b"# Design\nKeep A7."
        if source_type == "github_repo"
        else {
            "vault_id": "vault-a",
            "relative_path": "old/design.md",
            "file_lineage_id": "file-77",
            "markdown": "# Design\nKeep A7.",
        }
    )
    first_raw, first_normalized = _inputs(first_item, first_raw_value, "# Design\nKeep A7.")
    first = project_source_item(
        source_id=f"src-{source_type}",
        source_type=source_type,
        run_id="run-file-1",
        item=first_item,
        raw=first_raw,
        normalized=first_normalized,
    )
    moved_item = _item(item_id="file-new", extra=second_extra)
    moved_raw_value = (
        b"# Design\nKeep A7."
        if source_type == "github_repo"
        else {
            "vault_id": "vault-a",
            "relative_path": "new/design.md",
            "file_lineage_id": "file-77",
            "markdown": "# Design\nKeep A7.",
        }
    )
    moved_raw, moved_normalized = _inputs(moved_item, moved_raw_value, "# Design\nKeep A7.")
    moved = project_source_item(
        source_id=f"src-{source_type}",
        source_type=source_type,
        run_id="run-file-2",
        item=moved_item,
        raw=moved_raw,
        normalized=moved_normalized,
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions={
            first.observations[0].id: first.observation_revisions[0],
        },
    )

    assert moved.source_units[0].id == first.source_units[0].id
    assert moved.observations[0].id == first.observations[0].id
    assert moved.deltas[0].axes == frozenset({DeltaAxis.LOCATION})
    assert moved.deltas[0].requires_extraction is False


def test_attested_github_compare_previous_filename_preserves_unit_without_daemon_lineage() -> None:
    first_item = _item(
        item_id="file-old",
        extra={
            "relative_path": "old/design.md",
            "repo_owner": "acme",
            "repo_name": "pay",
            "repo_ref": "main",
        },
    )
    first_raw, first_normalized = _inputs(first_item, b"Keep A7.", "Keep A7.")
    first = project_source_item(
        source_id="src-github_repo",
        source_type="github_repo",
        run_id="run-compare-1",
        item=first_item,
        raw=first_raw,
        normalized=first_normalized,
    )
    moved_item = _item(
        item_id="file-new",
        extra={
            "relative_path": "new/design.md",
            "previous_filename": "old/design.md",
            "rename_evidence_authoritative": True,
            "repo_owner": "acme",
            "repo_name": "pay",
            "repo_ref": "main",
        },
    )
    moved_raw, moved_normalized = _inputs(moved_item, b"Keep A7.", "Keep A7.")

    moved = project_source_item(
        source_id="src-github_repo",
        source_type="github_repo",
        run_id="run-compare-2",
        item=moved_item,
        raw=moved_raw,
        normalized=moved_normalized,
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions={first.observations[0].id: first.observation_revisions[0]},
    )

    assert moved.source_units[0].id == first.source_units[0].id
    assert moved.deltas[0].axes == frozenset({DeltaAxis.LOCATION})

    ordinary_item = _item(
        item_id="file-new",
        extra={
            "relative_path": "new/design.md",
            "repo_owner": "acme",
            "repo_name": "pay",
            "repo_ref": "main",
        },
    )
    ordinary_raw, ordinary_normalized = _inputs(
        ordinary_item,
        b"Keep A7.",
        "Keep A7.",
    )
    ordinary = project_source_item(
        source_id="src-github_repo",
        source_type="github_repo",
        run_id="run-compare-3",
        item=ordinary_item,
        raw=ordinary_raw,
        normalized=ordinary_normalized,
        scope={
            "source_unit_id": moved.source_units[0].id,
            "source_unit_provider_key": moved.source_units[0].provider_key,
        },
        prior_unit_revision=moved.source_unit_revisions[0],
        prior_observation_revisions={moved.observations[0].id: moved.observation_revisions[0]},
    )

    assert ordinary.source_units[0].id == first.source_units[0].id
    assert ordinary.source_units[0].provider_key == first.source_units[0].provider_key
    assert ordinary.deltas[0].axes == frozenset()


def test_unattested_github_previous_filename_keeps_path_identity() -> None:
    item = _item(
        item_id="file-new",
        extra={
            "relative_path": "new/design.md",
            "previous_filename": "old/design.md",
            "repo_owner": "acme",
            "repo_name": "pay",
            "repo_ref": "main",
        },
    )
    raw, normalized = _inputs(item, b"Keep A7.", "Keep A7.")

    projection = project_source_item(
        source_id="src-github_repo",
        source_type="github_repo",
        run_id="run-unattested-rename",
        item=item,
        raw=raw,
        normalized=normalized,
    )

    assert projection.source_units[0].provider_key.endswith(":new/design.md")
    assert projection.relations == ()


def test_teams_window_is_unit_and_native_messages_are_observations() -> None:
    item = _item(
        item_id="window-1",
        extra={"conversation_id": "conv-1", "window_id": "window-1", "root_message_id": "msg-1"},
    )
    raw, normalized = _inputs(
        item,
        {
            "messages": [
                {"id": "msg-1", "content": "Question", "time": "2026-07-14T10:00:00Z"},
                {"id": "msg-2", "content": "Keep A7", "time": "2026-07-14T10:01:00Z"},
            ]
        },
    )
    projection = project_source_item(
        source_id="src-teams",
        source_type="teams",
        run_id="run-teams",
        item=item,
        raw=raw,
        normalized=normalized,
    )

    assert projection.source_units[0].provider_key == "window-1"
    assert projection.source_units[0].locator["observed_from"] == "2026-07-14T10:00:00+00:00"
    assert projection.source_units[0].locator["observed_to"] == "2026-07-14T10:01:00+00:00"
    assert [item.provider_key for item in projection.observations] == ["msg-1", "msg-2"]
    assert projection.coverage is ProjectionCoverage.PARTIAL_PROJECTION


def test_teams_window_attestation_proves_only_unit_snapshot_completeness() -> None:
    item = _item(
        item_id="window-1",
        extra={"conversation_id": "conv-1", "window_id": "window-1"},
    )
    raw, normalized = _inputs(
        item,
        {
            "_authoritative_snapshot": True,
                "messages": [
                    {"id": "msg-1", "content": "Question", "time": "2026-07-14T10:00:00Z"},
                    {"id": "msg-2", "content": "Answer", "time": "2026-07-14T10:01:00Z"},
            ],
        },
    )

    projection = project_source_item(
        source_id="src-teams",
        source_type="teams",
        run_id="run-teams-complete-window",
        item=item,
        raw=raw,
        normalized=normalized,
    )

    assert projection.coverage is ProjectionCoverage.COMPLETE_SNAPSHOT
    assert (
        source_run_projection_coverage(
            source_type="teams",
            incremental=False,
            authoritative_snapshot=False,
            discovery_complete=False,
        )
        is ProjectionCoverage.PARTIAL_PROJECTION
    )


def test_teams_bounded_time_scope_attestation_is_persisted_on_unit_locator() -> None:
    configured_scope = {
        "conversation_ids": ["conv-1"],
        "max_age_days": 30,
        "conversation_gap_minutes": 60,
        "max_block_messages": 100,
    }
    item = _item(
        item_id="window-1",
        extra={"conversation_id": "conv-1", "window_id": "window-1"},
    )
    raw, normalized = _inputs(
        item,
        {
            "_scope_coverage_from": "2026-07-01T00:00:00Z",
            "_scope_coverage_to": "2026-07-16T00:00:00Z",
            "messages": [
                {
                    "id": "msg-1",
                    "content": "Answer",
                    "time": "2026-07-10T09:00:00Z",
                }
            ],
        },
    )

    projection = project_source_item(
        source_id="src-teams",
        source_type="teams",
        run_id="run-teams-bounded-scope",
        item=item,
        raw=raw,
        normalized=normalized,
        scope={"configured_scope": configured_scope},
    )

    locator = projection.source_units[0].locator
    assert locator["time_scope_fingerprint"] == projection_scope_fingerprint(configured_scope)
    assert locator["time_scope_coverage_from"] == "2026-07-01T00:00:00+00:00"
    assert locator["time_scope_coverage_to"] == "2026-07-16T00:00:00+00:00"


def test_teams_complete_window_tombstone_removes_every_prior_observation() -> None:
    item = _item(
        item_id="window-1",
        extra={"conversation_id": "conv-1", "window_id": "window-1"},
    )
    first_raw, first_normalized = _inputs(
        item,
        {
            "_authoritative_snapshot": True,
                "messages": [
                    {"id": "msg-1", "content": "Question", "time": "2026-07-14T10:00:00Z"},
                    {"id": "msg-2", "content": "Answer", "time": "2026-07-14T10:01:00Z"},
            ],
        },
    )
    first = project_source_item(
        source_id="src-teams",
        source_type="teams",
        run_id="run-teams-before-tombstone",
        item=item,
        raw=first_raw,
        normalized=first_normalized,
    )
    tombstone_raw, tombstone_normalized = _inputs(
        item,
        {
            "_authoritative_snapshot": True,
            "_tombstone": True,
            "messages": [],
        },
        "",
    )

    tombstone = project_source_item(
        source_id="src-teams",
        source_type="teams",
        run_id="run-teams-tombstone",
        item=item,
        raw=tombstone_raw,
        normalized=tombstone_normalized,
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions={
            observation.id: revision
            for observation, revision in zip(
                first.observations,
                first.observation_revisions,
                strict=True,
            )
        },
    )

    assert tombstone.coverage is ProjectionCoverage.COMPLETE_SNAPSHOT
    assert tombstone.observations == ()
    assert set(tombstone.deltas[0].removed_observation_ids) == {observation.id for observation in first.observations}
    assert DeltaAxis.MEMBERSHIP in tombstone.deltas[0].axes


def test_teams_selector_scope_transition_closes_after_removed_units_are_tombstoned() -> None:
    adapter = GeneSourceProjectionAdapter()
    transition = ProjectionScopeTransition(
        id="transition-teams",
        source_id="src-teams",
        previous_scope={
            "conversation_ids": [
                "19:conv-a@example.test",
                "19:conv-b@example.test",
            ]
        },
        target_scope={"conversation_ids": ["19:conv-a@example.test"]},
    )
    retained = SourceUnit(
        id="unit-a",
        source_id="src-teams",
        unit_type="teams_window",
        provider_key="window-a",
        locator={
            "conversation_id": "19:conv-a@example.test",
            "window_id": "window-a",
        },
    )

    assert (
        adapter.reconciliation_coverage(
            source_type="teams",
            transition=transition,
            current_units=(retained,),
            run_attestations=_teams_run_attestations(transition),
        )
        is ProjectionCoverage.TOMBSTONED_DELTA
    )

    stale_removed_scope = SourceUnit(
        id="unit-b",
        source_id="src-teams",
        unit_type="teams_window",
        provider_key="window-b",
        locator={
            "conversation_id": "19:conv-b@example.test",
            "window_id": "window-b",
        },
    )
    assert (
        adapter.reconciliation_coverage(
            source_type="teams",
            transition=transition,
            current_units=(retained, stale_removed_scope),
            run_attestations=_teams_run_attestations(transition),
        )
        is None
    )


def test_teams_selector_scope_transition_rejects_unresolved_legacy_names() -> None:
    adapter = GeneSourceProjectionAdapter()
    transition = ProjectionScopeTransition(
        id="transition-teams-unresolved-selector",
        source_id="src-teams",
        previous_scope={"channels": ["Old Team/General"]},
        target_scope={"channels": ["New Team/General"]},
    )

    assert (
        adapter.reconciliation_coverage(
            source_type="teams",
            transition=transition,
            current_units=(),
        )
        is None
    )


def test_teams_empty_target_units_require_current_run_attestation() -> None:
    adapter = GeneSourceProjectionAdapter()
    transition = ProjectionScopeTransition(
        id="transition-teams-empty",
        source_id="src-teams",
        previous_scope={"conversation_ids": ["19:old@example.test"]},
        target_scope={"conversation_ids": ["19:empty@example.test"]},
    )

    assert (
        adapter.reconciliation_coverage(
            source_type="teams",
            transition=transition,
            current_units=(),
        )
        is None
    )
    assert (
        adapter.reconciliation_coverage(
            source_type="teams",
            transition=transition,
            current_units=(),
            run_attestations=_teams_run_attestations(transition),
        )
        is ProjectionCoverage.TOMBSTONED_DELTA
    )


def test_teams_time_scope_transition_requires_bounded_target_attestation() -> None:
    adapter = GeneSourceProjectionAdapter()
    target_scope = {
        "conversation_ids": ["19:conv-a@example.test"],
        "max_age_days": 30,
    }
    transition = ProjectionScopeTransition(
        id="transition-teams-time",
        source_id="src-teams",
        previous_scope={
            "conversation_ids": ["19:conv-a@example.test"],
            "max_age_days": 365,
        },
        target_scope=target_scope,
    )
    fingerprint = projection_scope_fingerprint(target_scope)
    attested = SourceUnit(
        id="unit-a",
        source_id="src-teams",
        unit_type="teams_window",
        provider_key="window-a",
        locator={
            "conversation_id": "19:conv-a@example.test",
            "window_id": "window-a",
            "observed_to": "2026-07-10T09:30:00+00:00",
            "time_scope_fingerprint": fingerprint,
            "time_scope_coverage_from": "2026-07-01T00:00:00+00:00",
            "time_scope_coverage_to": "2026-07-16T00:00:00+00:00",
        },
    )

    assert (
        adapter.reconciliation_coverage(
            source_type="teams",
            transition=transition,
            current_units=(attested,),
            run_attestations=_teams_run_attestations(transition),
        )
        is ProjectionCoverage.TOMBSTONED_DELTA
    )
    assert (
        adapter.reconciliation_coverage(
            source_type="teams",
            transition=transition,
            current_units=(
                SourceUnit(
                    id="unit-stale",
                    source_id="src-teams",
                    unit_type="teams_window",
                    provider_key="window-stale",
                    locator={
                        "conversation_id": "19:conv-a@example.test",
                        "window_id": "window-stale",
                    },
                ),
            ),
            run_attestations=_teams_run_attestations(transition),
        )
        is None
    )


def test_teams_partition_scope_transition_requires_complete_target_attestation() -> None:
    adapter = GeneSourceProjectionAdapter()
    target_scope = {
        "conversation_ids": ["19:conv-a@example.test"],
        "conversation_gap_minutes": 30,
        "max_block_messages": 50,
    }
    transition = ProjectionScopeTransition(
        id="transition-teams-partition",
        source_id="src-teams",
        previous_scope={
            "conversation_ids": ["19:conv-a@example.test"],
            "conversation_gap_minutes": 60,
            "max_block_messages": 100,
        },
        target_scope=target_scope,
    )
    fingerprint = projection_scope_fingerprint(target_scope)
    complete = SourceUnit(
        id="unit-a",
        source_id="src-teams",
        unit_type="teams_window",
        provider_key="window-a",
        locator={
            "conversation_id": "19:conv-a@example.test",
            "window_id": "window-a",
            "projection_scope_fingerprint": fingerprint,
        },
    )

    assert (
        adapter.reconciliation_coverage(
            source_type="teams",
            transition=transition,
            current_units=(complete,),
            run_attestations=_teams_run_attestations(transition),
        )
        is ProjectionCoverage.TOMBSTONED_DELTA
    )
    partial = SourceUnit(
        id="unit-b",
        source_id="src-teams",
        unit_type="teams_window",
        provider_key="window-b",
        locator={
            "conversation_id": "19:conv-a@example.test",
            "window_id": "window-b",
            "time_scope_fingerprint": fingerprint,
        },
    )
    assert (
        adapter.reconciliation_coverage(
            source_type="teams",
            transition=transition,
            current_units=(partial,),
            run_attestations=_teams_run_attestations(transition),
        )
        is None
    )


def test_teams_mixed_time_and_partition_transition_requires_both_axes() -> None:
    adapter = GeneSourceProjectionAdapter()
    target_scope = {
        "conversation_ids": ["19:conv-a@example.test"],
        "max_age_days": 30,
        "conversation_gap_minutes": 30,
        "max_block_messages": 50,
    }
    transition = ProjectionScopeTransition(
        id="transition-teams-mixed",
        source_id="src-teams",
        previous_scope={
            "conversation_ids": ["19:conv-a@example.test"],
            "max_age_days": 365,
            "conversation_gap_minutes": 60,
            "max_block_messages": 100,
        },
        target_scope=target_scope,
    )
    unit = SourceUnit(
        id="unit-mixed",
        source_id="src-teams",
        unit_type="teams_window",
        provider_key="window-mixed",
        locator={
            "conversation_id": "19:conv-a@example.test",
            "window_id": "window-mixed",
            "observed_to": "2026-07-10T09:30:00+00:00",
            "partition_scope_fingerprint": projection_scope_fingerprint(
                {
                    "conversation_gap_minutes": 30,
                    "max_block_messages": 50,
                }
            ),
            "time_scope_fingerprint": projection_scope_fingerprint(target_scope),
            "time_scope_coverage_from": "2026-07-01T00:00:00+00:00",
            "time_scope_coverage_to": "2026-07-16T00:00:00+00:00",
        },
    )

    assert (
        adapter.reconciliation_coverage(
            source_type="teams",
            transition=transition,
            current_units=(unit,),
            run_attestations=_teams_run_attestations(transition),
        )
        is ProjectionCoverage.TOMBSTONED_DELTA
    )


@pytest.mark.parametrize(
    ("source_type", "incremental", "authoritative_snapshot", "discovery_complete", "expected"),
    [
        ("confluence", False, False, True, ProjectionCoverage.COMPLETE_SNAPSHOT),
        ("jira", False, False, True, ProjectionCoverage.COMPLETE_SNAPSHOT),
        ("github_repo", False, False, True, ProjectionCoverage.COMPLETE_SNAPSHOT),
        ("github_pages", False, False, True, ProjectionCoverage.COMPLETE_SNAPSHOT),
        ("local_markdown", False, False, True, ProjectionCoverage.COMPLETE_SNAPSHOT),
        ("confluence", False, False, False, ProjectionCoverage.PARTIAL_PROJECTION),
        ("teams", False, False, False, ProjectionCoverage.PARTIAL_PROJECTION),
        ("agent_session", False, False, False, ProjectionCoverage.PARTIAL_PROJECTION),
        ("confluence", True, False, True, ProjectionCoverage.PARTIAL_PROJECTION),
        ("teams", True, True, False, ProjectionCoverage.COMPLETE_SNAPSHOT),
    ],
)
def test_run_coverage_only_proves_absence_for_authoritative_discovery(
    source_type: str,
    incremental: bool,
    authoritative_snapshot: bool,
    discovery_complete: bool,
    expected: ProjectionCoverage,
) -> None:
    assert (
        source_run_projection_coverage(
            source_type=source_type,
            incremental=incremental,
            authoritative_snapshot=authoritative_snapshot,
            discovery_complete=discovery_complete,
        )
        is expected
    )


def test_unit_tombstone_removes_all_prior_observations_with_explicit_coverage() -> None:
    item = _item(
        item_id="jira-PAY-12",
        extra={"issue_key": "PAY-12"},
    )
    raw, normalized = _inputs(
        item,
        _jira_payload(comments=[{"id": "501", "body": "Keep A7"}]),
    )
    initial = project_source_item(
        source_id="src-j",
        source_type="jira",
        run_id="run-before-delete",
        item=item,
        raw=raw,
        normalized=normalized,
    )

    tombstone = project_source_unit_tombstone(
        source_type="jira",
        run_id="run-delete",
        source_unit=initial.source_units[0],
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={revision.observation_id: revision for revision in initial.observation_revisions},
        reason="not_returned_by_authoritative_snapshot",
    )

    assert tombstone.coverage is ProjectionCoverage.TOMBSTONED_DELTA
    assert tombstone.observations == ()
    assert tombstone.source_unit_revisions[0].observation_revision_ids == ()
    assert tombstone.deltas[0].removed_observation_ids == tuple(
        sorted(observation.id for observation in initial.observations)
    )


def test_every_builtin_gene_has_an_explicit_projection_contract() -> None:
    assert set(GENE_REGISTRY) == set(BUILTIN_SPECIALIZED_SOURCE_TYPES)
