from __future__ import annotations

import json

from memforge.local_agent.teams_audit import (
    redact_teams_audit_event,
    validate_teams_audit_run,
    write_teams_audit_event,
)


def test_redact_teams_audit_event_removes_sensitive_fields_and_hashes_raw_ids():
    event = {
        "event": "teams_conversation_poll",
        "run_id": "run-1",
        "source_id": "src-teams",
        "raw_conversation_id": "19:real-channel-id@thread.tacv2",
        "conversation_key": ("src-teams", "19:real-channel-id@thread.tacv2"),
        "bearer_token": "secret-token",
        "raw_message_body": "real Teams message text",
        "participant_display_names": ["Alice", "Bob"],
        "metadata_sync_state": "opaque-sync-state",
        "metadata_backward_link": "https://teams.cloud.microsoft/api/chatsvc/emea/v1/users/ME/conversations/...",
    }

    redacted = redact_teams_audit_event(event)

    assert "raw_conversation_id" not in redacted
    assert "conversation_key" not in redacted
    assert "bearer_token" not in redacted
    assert "raw_message_body" not in redacted
    assert "participant_display_names" not in redacted
    assert "metadata_sync_state" not in redacted
    assert "metadata_backward_link" not in redacted
    assert redacted["raw_conversation_id_hash"].startswith("sha256:")
    assert redacted["conversation_key_hash"].startswith("sha256:")
    assert redacted["metadata_sync_state_hash"].startswith("sha256:")
    assert redacted["metadata_backward_link_hash"].startswith("sha256:")
    assert "real-channel-id" not in json.dumps(redacted)
    assert "real Teams message text" not in json.dumps(redacted)


def test_validate_teams_audit_run_accepts_complete_incremental_run():
    events = [
        {
            "event": "teams_sync_run",
            "run_id": "run-1",
            "source_id": "src-teams",
            "status": "completed",
            "claim_add": 1,
            "claim_update": 1,
            "claim_supersede": 0,
            "claim_noop": 2,
            "claim_rejected_ambiguous": 0,
        },
        {
            "event": "teams_conversation_poll",
            "run_id": "run-1",
            "pagination_complete": True,
            "access_probe_status": "ok",
            "covered_created_from": "2026-07-08T00:00:00+00:00",
            "covered_created_to": "2026-07-08T01:00:00+00:00",
            "raw_messages_seen": 10,
            "unique_message_keys_seen": 9,
            "duplicate_raw_messages": 1,
            "upsert_new": 2,
            "upsert_updated": 1,
            "upsert_unchanged": 5,
            "explicit_delete_markers": 1,
            "missing_once_candidates": 0,
        },
        {
            "event": "teams_window_projection",
            "run_id": "run-1",
            "window_id_hash": "win-1",
            "revision_hash": "rev-1",
            "receipt_status": "new",
        },
        {
            "event": "teams_window_projection",
            "run_id": "run-1",
            "window_id_hash": "win-2",
            "revision_hash": "rev-2",
            "receipt_status": "existing",
            "receipt_skip_reason": "receipt_exists",
        },
        {
            "event": "teams_memory_patch",
            "run_id": "run-1",
            "window_id_hash": "win-1",
            "revision_hash": "rev-1",
            "claim_add": 1,
            "claim_update": 1,
            "claim_supersede": 0,
            "claim_noop": 2,
            "claim_rejected_ambiguous": 0,
        },
    ]

    assert validate_teams_audit_run(events) == []


def test_validate_teams_audit_run_reports_duplicate_projection_and_missing_patch():
    events = [
        {
            "event": "teams_conversation_poll",
            "run_id": "run-1",
            "pagination_complete": True,
            "access_probe_status": "ok",
            "covered_created_from": "2026-07-08T00:00:00+00:00",
            "covered_created_to": "2026-07-08T01:00:00+00:00",
            "raw_messages_seen": 2,
            "unique_message_keys_seen": 2,
            "duplicate_raw_messages": 0,
            "upsert_new": 1,
            "upsert_updated": 0,
            "upsert_unchanged": 0,
            "explicit_delete_markers": 0,
            "missing_once_candidates": 0,
        },
        {
            "event": "teams_window_projection",
            "run_id": "run-1",
            "window_id_hash": "win-1",
            "revision_hash": "rev-1",
            "receipt_status": "new",
        },
        {
            "event": "teams_window_projection",
            "run_id": "run-1",
            "window_id_hash": "win-1",
            "revision_hash": "rev-1",
            "receipt_status": "new",
        },
    ]

    errors = validate_teams_audit_run(events)

    assert "poll ledger action counts do not reconcile for run run-1" in errors
    assert "duplicate new projection for run run-1 window win-1 revision rev-1" in errors
    assert "new projection missing one matching memory patch for run run-1 window win-1 revision rev-1" in errors


def test_write_teams_audit_event_appends_redacted_jsonl(tmp_path):
    audit_path = tmp_path / "teams-audit.jsonl"

    write_teams_audit_event(
        audit_path,
        {
            "event": "teams_conversation_poll",
            "run_id": "run-1",
            "raw_conversation_id": "19:secret-channel@thread.tacv2",
            "raw_message_body": "do not write this",
        },
    )

    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["raw_conversation_id_hash"].startswith("sha256:")
    assert "secret-channel" not in audit_path.read_text(encoding="utf-8")
    assert "do not write this" not in audit_path.read_text(encoding="utf-8")
