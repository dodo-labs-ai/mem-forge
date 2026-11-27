from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import timedelta

import pytest

from memforge.local_agent.source_contract import (
    DOCUMENT_ACQUISITION_OPERATION,
    LOCAL_AGENT_SYNC_OPERATIONS,
    SourceSyncRunReceiptError,
    execution_owner_user_id,
    is_local_agent_backed_source,
    local_agent_collection_attempt_id,
    local_agent_completion_status,
    local_agent_retry_delay,
    local_agent_job_config,
    local_agent_source_config_revision,
    local_agent_sync_job_payload,
    local_agent_sync_operation,
    local_agent_sync_snapshot_id,
    source_processing_receipt,
    source_sync_run_id_from_completion,
    source_execution_descriptor,
)
from memforge.server.source_admin_service import source_ownership_and_capabilities


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
            "document_acquisition",
            "teams_sync",
        }
    )


def test_local_agent_attempt_identity_and_retry_status_are_deterministic() -> None:
    assert local_agent_sync_snapshot_id("job-1", 2) == "job-1:attempt:2"
    assert local_agent_completion_status("failed", retryable=True, attempt_count=4) == "queued"
    assert local_agent_completion_status("failed", retryable=True, attempt_count=5) == "failed"
    assert local_agent_collection_attempt_id("teams", "job-1", 2, "job-1:attempt:2") == "job-1:attempt:2"
    assert local_agent_collection_attempt_id("teams", "job-1", 2) == "job-1:attempt:2"
    with pytest.raises(ValueError, match="does not match"):
        local_agent_collection_attempt_id("teams", "job-1", 2, "spoofed")
    with pytest.raises(ValueError, match="not collected by a local agent"):
        local_agent_collection_attempt_id("unregistered", "job-1", 2)
    # A document Source runs on a device through its executor, not a local-agent collection.
    with pytest.raises(ValueError, match="not collected by a local agent"):
        local_agent_collection_attempt_id("local_markdown", "job-1", 2)


def test_local_agent_retry_delay_uses_long_bounded_schedule() -> None:
    assert local_agent_retry_delay("failed", retryable=True, attempt_count=1) == timedelta(hours=1)
    assert local_agent_retry_delay("failed", retryable=True, attempt_count=2) == timedelta(hours=12)
    assert local_agent_retry_delay("failed", retryable=True, attempt_count=4) == timedelta(hours=12)
    assert local_agent_retry_delay("failed", retryable=True, attempt_count=5) is None
    assert local_agent_retry_delay("failed", retryable=False, attempt_count=1) is None
    assert local_agent_retry_delay("succeeded", retryable=True, attempt_count=1) is None


class _NoTransitions:
    """A store holding no open scope transition."""

    async def get_open_projection_scope_transition(self, source_id):
        return None


def test_local_agent_job_payload_preserves_complete_collection_scope() -> None:
    teams_source = {
        "id": "src-teams",
        "type": "teams",
        "config": {
            "conversation_ids": ["19:chat@example.test"],
            "max_age_days": 45,
            "max_block_messages": 250,
        },
    }

    teams = asyncio.run(local_agent_sync_job_payload(_NoTransitions(), teams_source))

    assert teams["initial_history_days"] == 45
    assert teams["rolling_retention_days"] is None
    assert "max_age_days" not in teams
    assert "max_block_messages" not in teams
    assert teams["source_config_revision"] == local_agent_source_config_revision(teams_source)


@pytest.mark.parametrize(
    ("source_type", "config", "expected_operation"),
    [
        ("teams", {}, "teams_sync"),
        ("local_markdown", {}, None),
        ("jira", {}, None),
        ("github_repo", {}, None),
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
            "teams",
            {"conversation_ids": ["19:chat@example.test"]},
            {"kind": "local_agent", "operation": "teams_sync", "immutable_config_fields": []},
        ),
        (
            "jira",
            {"base_url": "https://jira.example.test"},
            {"kind": "server", "operation": None, "immutable_config_fields": []},
        ),
        (
            "confluence",
            {},
            {"kind": "server", "operation": None, "immutable_config_fields": []},
        ),
    ],
)
def test_source_execution_descriptor_is_canonical_ui_contract(
    source_type: str,
    config: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert source_execution_descriptor({"type": source_type, "config": config}) == expected


def test_a_device_source_is_collected_by_device_acquisition() -> None:
    source = {"type": "confluence", "config": {"base_url": "https://wiki.example.com"}, "executor": "device"}

    assert source_execution_descriptor(source) == {
        "kind": "local_agent",
        "operation": DOCUMENT_ACQUISITION_OPERATION,
        "immutable_config_fields": [],
    }


def test_local_agent_source_contract_accepts_serialized_config() -> None:
    source = {
        "type": "teams",
        "config": json.dumps({"conversation_ids": ["19:chat@example.test"]}),
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
            "teams",
            {
                "region": "emea",
                "conversation_ids": ["19:team@example.test"],
                "conversation_gap_minutes": 60,
                "documents_dir": "/server/private",
                "password": "secret",
            },
            {
                "region": "emea",
                "conversation_ids": ["19:team@example.test"],
                "initial_history_days": 14,
                "rolling_retention_days": None,
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

    payload = asyncio.run(local_agent_sync_job_payload(
        _NoTransitions(),
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
    ))

    revision = payload.pop("source_config_revision")
    assert revision
    assert payload == {
        "region": "emea",
        "conversation_ids": ["19:canonical@example.test"],
        "initial_history_days": 14,
        "rolling_retention_days": None,
        "source_id": "src-teams",
        "source_type": "teams",
        "force_full_sync": True,
    }


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
        "type": "teams",
        "config": {"conversation_ids": ["19:chat@example.test"]},
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


