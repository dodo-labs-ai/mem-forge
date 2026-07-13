from __future__ import annotations

import json

import pytest

from memforge.genes.local_adapter_packages import has_package_manifest
from memforge.local_agent.source_contract import (
    LOCAL_AGENT_SYNC_OPERATIONS,
    execution_owner_user_id,
    is_local_agent_backed_source,
    local_agent_authoritative_snapshot_id,
    local_agent_completion_status,
    local_agent_input_sha256,
    local_agent_job_config,
    local_agent_source_config_revision,
    local_agent_sync_job_payload,
    local_agent_sync_operation,
    local_agent_sync_snapshot_id,
    source_execution_descriptor,
    source_with_sync_inputs,
)
from memforge.server.source_admin_service import source_ownership_and_capabilities


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
    assert local_agent_authoritative_snapshot_id("jira", "job-1", 2, "job-1:attempt:2") == "job-1:attempt:2"
    with pytest.raises(ValueError, match="does not match"):
        local_agent_authoritative_snapshot_id("jira", "job-1", 2, "spoofed")
    assert local_agent_authoritative_snapshot_id("teams", "job-1", 2) is None


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


def test_local_agent_input_identity_uses_document_and_content_identity() -> None:
    first = local_agent_input_sha256("doc-a", "content-v1")
    assert first == local_agent_input_sha256("doc-a", "content-v1")
    assert first != local_agent_input_sha256("doc-a", "content-v2")
    assert first != local_agent_input_sha256("doc-b", "content-v1")


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
