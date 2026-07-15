from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.pipeline.source_projection_adapters import (
    BUILTIN_SPECIALIZED_SOURCE_TYPES,
    project_source_item,
    project_source_unit_tombstone,
    source_run_projection_coverage,
)
from memforge.genes import GENE_REGISTRY
from memforge.source_projection import DeltaAxis, ProjectionCoverage, SourceRelationType


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
        {
            "id": "10012",
            "key": "PAY-12",
            "fields": {"summary": "Payroll", "description": "Context", "status": {"name": "Done"}},
            "_comments": [
                {"id": "501", "body": "First short reply", "created": "2026-07-14T10:00:00Z"},
                {"id": "502", "body": "Correction: retain A7", "created": "2026-07-14T10:01:00Z"},
            ],
        },
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
        {"key": "PAY-12", "fields": {"summary": "Payroll"}},
    )

    with pytest.raises(ValueError, match="immutable numeric issue id"):
        project_source_item(
            source_id="src-j",
            source_type="jira",
            run_id="run-j-missing-id",
            item=item,
            raw=raw,
            normalized=normalized,
        )


def test_truncated_jira_comments_force_partial_coverage() -> None:
    item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
    raw, normalized = _inputs(
        item,
        {
            "id": "10012",
            "key": "PAY-12",
            "fields": {"summary": "Payroll"},
            "_comments": [],
            "_comments_truncated": {"returned": 0, "total": 10},
        },
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
        {
            "id": "10012",
            "key": "PAY-12",
            "fields": {"summary": "Payroll"},
            "_comments": [{"id": "501", "body": "Keep A7"}],
        },
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
        {
            "id": "10012",
            "key": "PAY-12",
            "fields": {"summary": "Payroll updated"},
            "_comments": [],
            "_comments_truncated": {"returned": 0, "total": 1},
        },
    )
    partial = project_source_item(
        source_id="src-j",
        source_type="jira",
        run_id="run-j-partial",
        item=item,
        raw=partial_raw,
        normalized=partial_normalized,
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision
            for revision in first.observation_revisions
        },
    )

    prior_comment = next(
        revision
        for revision in first.observation_revisions
        if revision.observation_id != first.observations[0].id
    )
    carried = next(
        revision
        for revision in partial.observation_revisions
        if revision.observation_id == prior_comment.observation_id
    )
    assert partial.coverage is ProjectionCoverage.PARTIAL_PROJECTION
    assert carried.id == prior_comment.id
    assert carried.metadata["carried_forward"] is True
    assert partial.deltas[0].removed_observation_ids == ()


def test_incomplete_embedded_jira_changelog_forces_partial_coverage() -> None:
    item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
    raw, normalized = _inputs(
        item,
        {
            "id": "10012",
            "key": "PAY-12",
            "fields": {"summary": "Payroll"},
            "changelog": {"histories": [{"id": "1"}], "total": 2},
        },
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
            "raw_payload": {
                "id": "10012",
                "key": "PAY-12",
                "fields": {"summary": "Payroll"},
                "changelog": {"histories": [{"id": "1"}], "total": 2},
            },
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


@pytest.mark.parametrize("source_type", ["jira", "teams"])
def test_operational_message_metadata_does_not_create_semantic_revision(source_type: str) -> None:
    if source_type == "jira":
        item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})
        first_payload = {
            "id": "10012",
            "key": "PAY-12",
            "fields": {"summary": "Payroll"},
            "_comments": [{"id": "501", "body": "Keep A7", "updated": "2026-07-14T10:00:00Z"}],
        }
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
            "messages": [
                {"id": "msg-1", "content": "Keep A7", "lastModifiedDateTime": "2026-07-14T10:00:00Z"}
            ]
        }
        second_payload = {
            "messages": [
                {
                    "id": "msg-1",
                    "content": "Keep A7",
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


def test_github_compare_previous_filename_preserves_unit_without_daemon_lineage() -> None:
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
        prior_observation_revisions={
            first.observations[0].id: first.observation_revisions[0]
        },
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
        prior_observation_revisions={
            moved.observations[0].id: moved.observation_revisions[0]
        },
    )

    assert ordinary.source_units[0].id == first.source_units[0].id
    assert ordinary.source_units[0].provider_key == first.source_units[0].provider_key
    assert ordinary.deltas[0].axes == frozenset()


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
    assert [item.provider_key for item in projection.observations] == ["msg-1", "msg-2"]
    assert projection.coverage is ProjectionCoverage.PARTIAL_PROJECTION


@pytest.mark.parametrize(
    ("source_type", "incremental", "authoritative_snapshot", "expected"),
    [
        ("confluence", False, False, ProjectionCoverage.COMPLETE_SNAPSHOT),
        ("jira", False, False, ProjectionCoverage.COMPLETE_SNAPSHOT),
        ("github_repo", False, False, ProjectionCoverage.COMPLETE_SNAPSHOT),
        ("github_pages", False, False, ProjectionCoverage.COMPLETE_SNAPSHOT),
        ("local_markdown", False, False, ProjectionCoverage.COMPLETE_SNAPSHOT),
        ("teams", False, False, ProjectionCoverage.PARTIAL_PROJECTION),
        ("agent_session", False, False, ProjectionCoverage.PARTIAL_PROJECTION),
        ("confluence", True, False, ProjectionCoverage.PARTIAL_PROJECTION),
        ("teams", True, True, ProjectionCoverage.COMPLETE_SNAPSHOT),
    ],
)
def test_run_coverage_only_proves_absence_for_authoritative_discovery(
    source_type: str,
    incremental: bool,
    authoritative_snapshot: bool,
    expected: ProjectionCoverage,
) -> None:
    assert source_run_projection_coverage(
        source_type=source_type,
        incremental=incremental,
        authoritative_snapshot=authoritative_snapshot,
    ) is expected


def test_unit_tombstone_removes_all_prior_observations_with_explicit_coverage() -> None:
    item = _item(
        item_id="jira-PAY-12",
        extra={"issue_key": "PAY-12"},
    )
    raw, normalized = _inputs(
        item,
        {
            "id": "10012",
            "key": "PAY-12",
            "fields": {"summary": "Payroll"},
            "_comments": [{"id": "501", "body": "Keep A7"}],
        },
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
        prior_observation_revisions={
            revision.observation_id: revision for revision in initial.observation_revisions
        },
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
