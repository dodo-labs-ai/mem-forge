from __future__ import annotations

import hashlib
import json

import pytest

from memforge.genes.local_adapter_packages import has_package_manifest
from memforge.local_agent.source_contract import (
    LOCAL_AGENT_SYNC_OPERATIONS,
    SourceSyncRunReceiptError,
    execution_owner_user_id,
    is_local_agent_backed_source,
    local_agent_collection_attempt_id,
    local_agent_completion_status,
    local_agent_input_sha256,
    local_agent_job_config,
    local_agent_rebaseline_snapshot_is_authoritative,
    local_agent_semantic_input_sha256,
    local_agent_source_config_revision,
    local_agent_sync_job_payload,
    local_agent_sync_operation,
    local_agent_sync_snapshot_id,
    source_processing_receipt,
    source_sync_run_id_from_completion,
    source_execution_descriptor,
    source_with_sync_inputs,
    validate_local_agent_replay_package,
)
from memforge.local_agent.replay_adapter import (
    get_local_source_replay_adapter,
    registered_local_source_types,
)
from memforge.server.source_admin_service import source_ownership_and_capabilities
from memforge.models import content_hash
from memforge.local_agent.teams_ledger import build_teams_window_id


def _canonical_payload_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def test_source_sync_run_receipt_is_required_on_both_sides_of_broker_handoff() -> None:
    assert source_processing_receipt({"run_id": "run-1"}) == {"source_sync_run_id": "run-1"}
    assert source_sync_run_id_from_completion({"source_sync_run_id": "run-1"}) == "run-1"
    assert source_processing_receipt({"error": "temporarily unavailable"}) == {}

    with pytest.raises(SourceSyncRunReceiptError, match="omitted run_id"):
        source_processing_receipt({})
    with pytest.raises(SourceSyncRunReceiptError, match="source_sync_run_id"):
        source_sync_run_id_from_completion({})


def _valid_jira_raw(*, summary: str = "Issue", comments: list[dict] | None = None) -> dict:
    comments = list(comments or [])
    return {
        "id": "10001",
        "key": "PAY-1",
        "fields": {
            "summary": summary,
            "description": None,
            "status": None,
            "priority": None,
            "assignee": None,
            "labels": [],
            "resolution": None,
            "updated": "2026-07-16T09:00:00+00:00",
        },
        "_comments": comments,
        "_comments_included": True,
        "_comments_total": len(comments),
        "changelog": {"startAt": 0, "histories": [], "total": 0},
    }


def test_local_agent_sync_operations_are_exported_from_the_domain_contract() -> None:
    assert LOCAL_AGENT_SYNC_OPERATIONS == frozenset(
        {
            "github_repo_sync",
            "jira_sync",
            "local_markdown_sync",
            "teams_sync",
        }
    )


def test_local_agent_attempt_identity_and_retry_status_are_deterministic() -> None:
    assert local_agent_sync_snapshot_id("job-1", 2) == "job-1:attempt:2"
    assert local_agent_completion_status("failed", retryable=True, attempt_count=4) == "queued"
    assert local_agent_completion_status("failed", retryable=True, attempt_count=5) == "failed"
    assert local_agent_collection_attempt_id("jira", "job-1", 2, "job-1:attempt:2") == "job-1:attempt:2"
    assert local_agent_collection_attempt_id("teams", "job-1", 2) == "job-1:attempt:2"
    with pytest.raises(ValueError, match="does not match"):
        local_agent_collection_attempt_id("jira", "job-1", 2, "spoofed")
    with pytest.raises(ValueError, match="not registered"):
        local_agent_collection_attempt_id("unregistered", "job-1", 2)


@pytest.mark.parametrize("source_type", ["github_repo", "jira", "local_markdown"])
def test_document_collection_snapshot_is_authoritative_for_rebaseline(
    source_type: str,
) -> None:
    assert local_agent_rebaseline_snapshot_is_authoritative(
        source_type,
        force_full_sync=False,
        input_snapshot_id="job-1:attempt:1",
    )


def test_teams_snapshot_is_authoritative_for_rebaseline_only_when_force_full() -> None:
    assert local_agent_rebaseline_snapshot_is_authoritative(
        "teams",
        force_full_sync=True,
        input_snapshot_id="job-1:attempt:1",
    )
    assert not local_agent_rebaseline_snapshot_is_authoritative(
        "teams",
        force_full_sync=False,
        input_snapshot_id="job-1:attempt:1",
    )
    assert not local_agent_rebaseline_snapshot_is_authoritative(
        "teams",
        force_full_sync=True,
        input_snapshot_id=None,
    )


def test_unregistered_source_never_inherits_rebaseline_snapshot_authority() -> None:
    assert not local_agent_rebaseline_snapshot_is_authoritative(
        "unregistered",
        force_full_sync=True,
        input_snapshot_id="job-1:attempt:1",
    )


def test_local_agent_job_payload_preserves_complete_collection_scope() -> None:
    jira_source = {
        "id": "src-jira",
        "type": "jira",
        "config": {
            "base_url": "https://jira.example.test",
            "query_mode": "advanced",
            "jql": "project = PAY",
            "jql_filter": "updated >= -90d",
        },
    }
    teams_source = {
        "id": "src-teams",
        "type": "teams",
        "config": {
            "conversation_ids": ["19:chat@example.test"],
            "max_age_days": 45,
            "max_block_messages": 250,
        },
    }

    jira = local_agent_sync_job_payload(jira_source)
    teams = local_agent_sync_job_payload(teams_source)

    assert jira["query_mode"] == "advanced"
    assert jira["jql_filter"] == "updated >= -90d"
    assert teams["max_age_days"] == 45
    assert teams["max_block_messages"] == 250
    assert jira["source_config_revision"] == local_agent_source_config_revision(jira_source)
    assert teams["source_config_revision"] == local_agent_source_config_revision(teams_source)


def test_local_replay_adapter_registry_has_no_provider_fallback() -> None:
    assert registered_local_source_types() == frozenset({"github_repo", "jira", "local_markdown", "teams"})
    with pytest.raises(
        ValueError,
        match="local source replay adapter is not registered: unknown",
    ):
        get_local_source_replay_adapter("unknown")


def test_local_agent_input_identity_uses_document_and_content_identity() -> None:
    first = local_agent_input_sha256("doc-a", "content-v1")
    assert first == local_agent_input_sha256("doc-a", "content-v1")
    assert first != local_agent_input_sha256("doc-a", "content-v2")
    assert first != local_agent_input_sha256("doc-b", "content-v1")


@pytest.mark.parametrize(
    ("source_type", "package", "missing_payload_field"),
    [
        (
            "github_repo",
            {
                "package_kind": "github_repo_document",
                "doc_id": "doc-a",
                "version": content_hash("# Repository"),
                "markdown": "# Repository",
            },
            "markdown",
        ),
        (
            "jira",
            {
                "package_kind": "jira_document",
                "doc_id": "doc-a",
                "version": "version-a",
                "raw_hash": "version-a",
                "issue_key": "PAY-1",
                "raw_payload": _valid_jira_raw(),
            },
            "raw_payload",
        ),
        (
            "local_markdown",
            {
                "package_kind": "local_markdown_document",
                "doc_id": "doc-a",
                "version": content_hash("# Local"),
                "markdown": "# Local",
            },
            "markdown",
        ),
        (
            "teams",
            {
                "package_kind": "teams_window_document",
                "doc_id": "doc-a",
                "version": "version-a",
                "revision_hash": "version-a",
                "raw_hash": "payload-hash",
                "conversation_id": "19:chat-a@example.test",
                "window_id": "window-a",
                "raw_payload": {
                    "conversation_id": "19:chat-a@example.test",
                    "window_id": "window-a",
                    "messages": [
                        {
                            "id": "message-a",
                            "content": "Decision",
                            "time": "2026-07-16T09:00:00+00:00",
                        }
                    ],
                },
            },
            "raw_payload",
        ),
    ],
)
def test_replay_package_identity_validation_covers_every_local_source_type(
    source_type: str,
    package: dict[str, object],
    missing_payload_field: str,
) -> None:
    if source_type == "teams":
        window_id = build_teams_window_id(
            source_id="src-teams",
            conversation_id="19:chat-a@example.test",
            root_or_anchor_message_id="message-a",
            window_type="time_block",
        )
        package["window_id"] = window_id
        package["root_message_id"] = "message-a"
        package["window_type"] = "time_block"
        package["raw_payload"]["window_id"] = window_id
    if source_type in {"jira", "teams"}:
        semantic_hash = _canonical_payload_hash(package["raw_payload"])
        package["semantic_hash"] = semantic_hash
        if source_type == "jira":
            package["version"] = semantic_hash
    else:
        semantic_hash = content_hash(str(package["markdown"]))
    body = json.dumps(package).encode()
    package_hash = hashlib.sha256(body).hexdigest()
    expected_version = str(package["version"])

    validate_local_agent_replay_package(
        source_type,
        body,
        expected_doc_id="doc-a",
        expected_version=expected_version,
        expected_input_sha256=local_agent_semantic_input_sha256(
            "doc-a",
            semantic_hash,
        ),
        expected_package_sha256=package_hash,
    )

    with pytest.raises(ValueError, match="source_lifecycle_local_replay_artifact_invalid"):
        validate_local_agent_replay_package(
            source_type,
            body,
            expected_doc_id="doc-a",
            expected_version="wrong-version",
            expected_input_sha256=local_agent_semantic_input_sha256(
                "doc-a",
                semantic_hash,
            ),
            expected_package_sha256=package_hash,
        )

    incomplete = {**package}
    incomplete.pop(missing_payload_field)
    incomplete_body = json.dumps(incomplete).encode()
    with pytest.raises(ValueError, match="source_lifecycle_local_replay_artifact_invalid"):
        validate_local_agent_replay_package(
            source_type,
            incomplete_body,
            expected_doc_id="doc-a",
            expected_version=expected_version,
            expected_input_sha256=local_agent_semantic_input_sha256(
                "doc-a",
                semantic_hash,
            ),
            expected_package_sha256=hashlib.sha256(incomplete_body).hexdigest(),
        )


def test_teams_replay_rejects_message_without_stable_source_time() -> None:
    package = {
        "package_kind": "teams_window_document",
        "doc_id": "doc-teams",
        "version": "revision-a",
        "revision_hash": "revision-a",
        "raw_hash": "raw-a",
        "conversation_id": "19:chat-a@example.test",
        "window_id": "window-a",
        "raw_payload": {
            "conversation_id": "19:chat-a@example.test",
            "window_id": "window-a",
            "messages": [{"id": "message-a", "content": "Decision"}],
        },
    }
    semantic_hash = _canonical_payload_hash(package["raw_payload"])
    package["semantic_hash"] = semantic_hash
    body = json.dumps(package).encode()

    with pytest.raises(
        ValueError,
        match="source_lifecycle_local_replay_artifact_invalid",
    ):
        validate_local_agent_replay_package(
            "teams",
            body,
            expected_doc_id="doc-teams",
            expected_version="revision-a",
            expected_input_sha256=local_agent_semantic_input_sha256(
                "doc-teams",
                semantic_hash,
            ),
            expected_package_sha256=hashlib.sha256(body).hexdigest(),
        )


def test_jira_replay_rejects_comment_without_stable_provider_id() -> None:
    raw_payload = _valid_jira_raw(comments=[{"body": "Decision"}])
    semantic_hash = _canonical_payload_hash(raw_payload)
    package = {
        "package_kind": "jira_document",
        "doc_id": "doc-jira",
        "version": semantic_hash,
        "raw_hash": semantic_hash,
        "semantic_hash": semantic_hash,
        "issue_key": "PAY-1",
        "raw_payload": raw_payload,
    }
    body = json.dumps(package).encode()

    with pytest.raises(ValueError, match="source_lifecycle_local_replay_artifact_invalid"):
        validate_local_agent_replay_package(
            "jira",
            body,
            expected_doc_id="doc-jira",
            expected_version=semantic_hash,
            expected_input_sha256=local_agent_semantic_input_sha256("doc-jira", semantic_hash),
            expected_package_sha256=hashlib.sha256(body).hexdigest(),
        )


@pytest.mark.parametrize("source_type", ["github_repo", "local_markdown"])
def test_replay_package_rejects_markdown_that_no_longer_matches_content_version(
    source_type: str,
) -> None:
    package_kind = "github_repo_document" if source_type == "github_repo" else "local_markdown_document"
    version = content_hash("original")
    body = json.dumps(
        {
            "package_kind": package_kind,
            "doc_id": "doc-a",
            "version": version,
            "markdown": "tampered",
        }
    ).encode()

    with pytest.raises(ValueError, match="source_lifecycle_local_replay_artifact_invalid"):
        validate_local_agent_replay_package(
            source_type,
            body,
            expected_doc_id="doc-a",
            expected_version=version,
            expected_input_sha256=local_agent_semantic_input_sha256(
                "doc-a",
                content_hash("original"),
            ),
            expected_package_sha256=hashlib.sha256(body).hexdigest(),
        )


def test_replay_package_rejects_tampered_payload_and_package_bytes() -> None:
    raw_payload = _valid_jira_raw(summary="Original")
    semantic_hash = _canonical_payload_hash(raw_payload)
    package = {
        "package_kind": "jira_document",
        "doc_id": "doc-a",
        "version": semantic_hash,
        "raw_hash": semantic_hash,
        "semantic_hash": semantic_hash,
        "issue_key": "PAY-1",
        "raw_payload": raw_payload,
    }
    body = json.dumps(package, sort_keys=True).encode()
    package_hash = hashlib.sha256(body).hexdigest()

    validate_local_agent_replay_package(
        "jira",
        body,
        expected_doc_id="doc-a",
        expected_version=semantic_hash,
        expected_input_sha256=local_agent_semantic_input_sha256("doc-a", semantic_hash),
        expected_package_sha256=package_hash,
    )

    tampered = json.dumps(
        {
            **package,
            "raw_payload": {
                "key": "PAY-1",
                "fields": {"summary": "Tampered"},
            },
        },
        sort_keys=True,
    ).encode()
    with pytest.raises(ValueError, match="source_lifecycle_local_replay_artifact_invalid"):
        validate_local_agent_replay_package(
            "jira",
            tampered,
            expected_doc_id="doc-a",
            expected_version=semantic_hash,
            expected_input_sha256=local_agent_semantic_input_sha256(
                "doc-a",
                semantic_hash,
            ),
            expected_package_sha256=package_hash,
        )


def test_local_package_validation_fails_closed_without_byte_attestation() -> None:
    markdown = "# Durable package"
    semantic_hash = content_hash(markdown)
    body = json.dumps(
        {
            "package_kind": "local_markdown_document",
            "doc_id": "doc-a",
            "version": semantic_hash,
            "markdown": markdown,
        }
    ).encode()

    with pytest.raises(
        ValueError,
        match="source_lifecycle_local_replay_attestation_required",
    ):
        validate_local_agent_replay_package(
            "local_markdown",
            body,
            expected_doc_id="doc-a",
            expected_version=semantic_hash,
            expected_input_sha256=local_agent_semantic_input_sha256(
                "doc-a",
                semantic_hash,
            ),
            expected_package_sha256="",
        )


@pytest.mark.parametrize(
    ("source_type", "config", "expected_operation"),
    [
        ("teams", {}, "teams_sync"),
        ("jira", {"sync_mode": "local_agent"}, "jira_sync"),
        ("jira", {"sync_mode": "cloud"}, None),
        ("local_markdown", {}, "local_markdown_sync"),
        (
            "github_repo",
            {"connection_mode": "local_push"},
            "github_repo_sync",
        ),
        ("github_repo", {"connection_mode": "cloud_pull"}, None),
        ("confluence", {}, None),
    ],
)
def test_local_agent_sync_operation_classifies_execution_mode(
    source_type: str,
    config: dict[str, object],
    expected_operation: str | None,
) -> None:
    assert local_agent_sync_operation(source_type, config) == expected_operation
    assert is_local_agent_backed_source({"type": source_type, "config": config}) is (expected_operation is not None)


@pytest.mark.parametrize(
    ("source_type", "config", "expected"),
    [
        (
            "jira",
            {"sync_mode": "local_agent"},
            {
                "kind": "local_agent",
                "operation": "jira_sync",
                "immutable_config_fields": ["sync_mode"],
            },
        ),
        (
            "github_repo",
            {"connection_mode": "cloud_pull"},
            {
                "kind": "server",
                "operation": None,
                "immutable_config_fields": ["connection_mode"],
            },
        ),
        (
            "confluence",
            {},
            {
                "kind": "server",
                "operation": None,
                "immutable_config_fields": [],
            },
        ),
    ],
)
def test_source_execution_descriptor_is_canonical_ui_contract(
    source_type: str,
    config: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert source_execution_descriptor(source_type, config) == expected


def test_local_agent_source_contract_accepts_serialized_config() -> None:
    source = {
        "type": "jira",
        "config": json.dumps({"sync_mode": "local_agent"}),
    }

    assert is_local_agent_backed_source(source) is True


def test_local_agent_source_contract_rejects_malformed_config() -> None:
    source = {"type": "jira", "config": "not-json"}

    assert is_local_agent_backed_source(source) is False


def test_execution_owner_normalizes_missing_and_blank_values() -> None:
    assert execution_owner_user_id({}) is None
    assert execution_owner_user_id({"execution_owner_user_id": "  "}) is None
    assert execution_owner_user_id({"execution_owner_user_id": " owner-a "}) == "owner-a"


@pytest.mark.parametrize(
    ("source_type", "config", "expected"),
    [
        (
            "github_repo",
            {
                "connection_mode": "local_push",
                "repo_url": "https://github.example.test/org/repo",
                "ref": "main",
                "repo_path": "/Users/me/repo",
                "include_paths": ["docs/"],
                "exclude_paths": ["docs/archived/"],
                "documents_dir": "/server/private",
                "api_token": "secret",
            },
            {
                "repo_url": "https://github.example.test/org/repo",
                "ref": "main",
                "include_paths": ["docs/"],
                "exclude_paths": ["docs/archived/"],
            },
        ),
        (
            "jira",
            {
                "sync_mode": "local_agent",
                "base_url": "https://jira.example.test",
                "auth_mode": "browser_cookie",
                "projects": ["PAY"],
                "local_agent_documents_dir": "/server/private",
                "pat": "secret",
                "client_secret": "secret",
            },
            {
                "sync_mode": "local_agent",
                "base_url": "https://jira.example.test",
                "auth_mode": "browser_cookie",
                "projects": ["PAY"],
            },
        ),
        (
            "teams",
            {
                "region": "emea",
                "conversation_ids": ["19:team@example.test"],
                "conversation_gap_minutes": 60,
                "local_agent_package_manifest": [{"doc_id": "private"}],
                "password": "secret",
            },
            {
                "region": "emea",
                "conversation_ids": ["19:team@example.test"],
                "conversation_gap_minutes": 60,
            },
        ),
    ],
)
def test_local_agent_job_config_is_source_type_allowlisted(
    source_type: str,
    config: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert local_agent_job_config(source_type, config) == expected


def test_local_agent_sync_job_payload_uses_saved_config_and_request_controls_only() -> None:
    source = {
        "id": "src-teams",
        "type": "teams",
        "config": {
            "region": "emea",
            "conversation_ids": ["19:canonical@example.test"],
            "password": "must-not-travel",
        },
    }

    payload = local_agent_sync_job_payload(
        source,
        {
            "conversation_ids": ["19:spoofed@example.test"],
            "execution_owner_user_id": "attacker",
            "audit_log_path": "/tmp/server-selected-path",
            "submitted_by": "spoofed-user",
            "process_now": False,
            "force_full_sync": True,
            "limit": 25,
        },
    )

    revision = payload.pop("source_config_revision")
    assert revision
    assert payload == {
        "region": "emea",
        "conversation_ids": ["19:canonical@example.test"],
        "source_id": "src-teams",
        "source_type": "teams",
        "source_activity_epoch": 0,
        "force_full_sync": True,
    }


def test_source_with_sync_inputs_respects_explicit_empty_snapshot() -> None:
    projected = source_with_sync_inputs(
        {
            "id": "src-local",
            "type": "local_markdown",
            "config": {"documents_dir": "/server/inbox"},
        },
        [],
        authoritative_snapshot=True,
    )

    assert projected["config"]["local_agent_package_manifest"] == []


def test_explicit_empty_package_manifest_is_authoritative() -> None:
    assert has_package_manifest({"local_agent_package_manifest": []}) is True
    assert has_package_manifest({}) is False


def test_local_source_owner_receives_execution_capabilities() -> None:
    source = {
        "type": "teams",
        "config": {},
        "created_by_user_id": "owner-a",
        "access_policy": "private",
        "access_state": "active",
        "owner_user_id": "owner-a",
        "execution_owner_user_id": "owner-a",
    }

    ownership, capabilities = source_ownership_and_capabilities(
        source,
        viewer_id="owner-a",
        viewer_role="member",
    )

    assert ownership["execution_owner_user_id"] == "owner-a"
    assert capabilities["can_configure"] is True
    assert capabilities["can_configure_connection"] is True
    assert capabilities["can_sync"] is True
    assert capabilities["can_force_resync"] is True


def test_local_source_non_owner_admin_keeps_management_without_execution() -> None:
    source = {
        "type": "teams",
        "config": {},
        "created_by_user_id": "owner-a",
        "access_policy": "workspace",
        "access_state": "active",
        "owner_user_id": "owner-a",
        "execution_owner_user_id": "owner-a",
    }

    ownership, capabilities = source_ownership_and_capabilities(
        source,
        viewer_id="admin-b",
        viewer_role="workspace_admin",
    )

    assert ownership["execution_owner_user_id"] == "owner-a"
    assert capabilities["can_configure"] is True
    assert capabilities["can_delete"] is True
    assert capabilities["can_configure_connection"] is False
    assert capabilities["can_sync"] is False
    assert capabilities["can_force_resync"] is False


def test_local_source_without_execution_owner_cannot_execute() -> None:
    source = {
        "type": "local_markdown",
        "config": {"root": "/repo"},
        "created_by_user_id": "owner-a",
        "access_policy": "private",
        "access_state": "active",
        "owner_user_id": "owner-a",
        "execution_owner_user_id": None,
    }

    _, capabilities = source_ownership_and_capabilities(
        source,
        viewer_id="owner-a",
        viewer_role="member",
    )

    assert capabilities["can_configure"] is True
    assert capabilities["can_configure_connection"] is False
    assert capabilities["can_sync"] is False
    assert capabilities["can_force_resync"] is False


def test_server_source_admin_retains_existing_execution_capabilities() -> None:
    source = {
        "type": "confluence",
        "config": {},
        "created_by_user_id": "owner-a",
        "access_policy": "workspace",
        "access_state": "active",
        "owner_user_id": "owner-a",
        "execution_owner_user_id": None,
    }

    _, capabilities = source_ownership_and_capabilities(
        source,
        viewer_id="admin-b",
        viewer_role="workspace_admin",
    )

    assert capabilities["can_configure"] is True
    assert capabilities["can_configure_connection"] is True
    assert capabilities["can_sync"] is True
    assert capabilities["can_force_resync"] is True
