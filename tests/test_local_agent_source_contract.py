from __future__ import annotations

import json

import pytest

from memforge.local_agent.source_contract import (
    LOCAL_AGENT_SYNC_OPERATIONS,
    execution_owner_user_id,
    is_local_agent_backed_source,
    local_agent_sync_operation,
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
    assert is_local_agent_backed_source({"type": source_type, "config": config}) is (
        expected_operation is not None
    )


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


def test_local_source_owner_receives_execution_capabilities() -> None:
    source = {
        "type": "teams",
        "config": {},
        "created_by_user_id": "owner-a",
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
