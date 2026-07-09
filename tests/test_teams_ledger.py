from __future__ import annotations

from datetime import datetime, timezone

from memforge.local_agent.teams_ledger import (
    TeamsLedgerMessage,
    TeamsLedgerProjector,
    TeamsLedgerStateStore,
    build_teams_receipt_key,
    build_teams_window_id,
    decode_teams_window_id,
)


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _message(message_id: str, created_at: str, body: str = "hello") -> TeamsLedgerMessage:
    return TeamsLedgerMessage(
        source_id="src-teams",
        conversation_id="19:conversation@thread.tacv2",
        conversation_type="group_chat",
        message_id=message_id,
        created_at=_ts(created_at),
        body_normalized=body,
    )


def test_teams_window_id_is_opaque_and_round_trips_path_hostile_components():
    window_id = build_teams_window_id(
        source_id="src:teams",
        conversation_id="19:conversation@thread.tacv2",
        root_or_anchor_message_id="1783500000000",
        window_type="time_block",
    )

    assert window_id.startswith("teams-block:v1:")
    assert "19:conversation@thread.tacv2" not in window_id

    decoded = decode_teams_window_id(window_id)
    assert decoded == {
        "source_id": "src:teams",
        "conversation_id": "19:conversation@thread.tacv2",
        "root_or_anchor_message_id": "1783500000000",
        "window_type": "time_block",
    }


def test_group_chat_exact_sixty_minute_gap_stays_in_same_frozen_block():
    projector = TeamsLedgerProjector(gap_minutes=60)

    result = projector.project_unthreaded(
        [
            _message("m1", "2026-07-08T09:00:00"),
            _message("m2", "2026-07-08T10:00:00"),
            _message("m3", "2026-07-08T11:00:01"),
        ]
    )

    assert len(result.blocks) == 2
    first, second = result.blocks
    assert first.frozen_anchor_message_id == "m1"
    assert first.member_message_ids == ("m1", "m2")
    assert second.frozen_anchor_message_id == "m3"


def test_late_message_before_anchor_expands_bounds_without_changing_window_id():
    projector = TeamsLedgerProjector(gap_minutes=60)
    initial = projector.project_unthreaded(
        [
            _message("m2", "2026-07-08T10:00:00", "anchor"),
            _message("m3", "2026-07-08T10:30:00", "follow-up"),
        ]
    )
    original_block = initial.blocks[0]

    updated = projector.project_unthreaded(
        [
            _message("m1", "2026-07-08T09:30:00", "late earlier"),
            _message("m2", "2026-07-08T10:00:00", "anchor"),
            _message("m3", "2026-07-08T10:30:00", "follow-up"),
        ],
        previous=initial,
    )

    assert len(updated.blocks) == 1
    block = updated.blocks[0]
    assert block.window_id == original_block.window_id
    assert block.frozen_anchor_message_id == "m2"
    assert block.member_min_created_at == _ts("2026-07-08T09:30:00")
    assert block.member_max_created_at == _ts("2026-07-08T10:30:00")
    assert block.member_message_ids == ("m1", "m2", "m3")
    assert block.revision_hash != original_block.revision_hash


def test_receipt_key_uses_source_window_and_revision_without_policy_version():
    receipt = build_teams_receipt_key(
        source_id="src-teams",
        window_id="teams-block:v1:opaque",
        revision_hash="sha256:revision",
    )

    assert receipt == {
        "source_id": "src-teams",
        "window_id": "teams-block:v1:opaque",
        "revision_hash": "sha256:revision",
    }


def test_teams_ledger_state_store_preserves_frozen_block_anchor_across_restart(tmp_path):
    state_path = tmp_path / "teams-ledger.json"
    projector = TeamsLedgerProjector(gap_minutes=60)
    initial = projector.project_unthreaded(
        [
            _message("m2", "2026-07-08T10:00:00", "anchor"),
            _message("m3", "2026-07-08T10:30:00", "follow-up"),
        ]
    )
    original_block = initial.blocks[0]

    TeamsLedgerStateStore(state_path).save_projection(
        source_id="src-teams",
        conversation_id="19:conversation@thread.tacv2",
        projection=initial,
    )

    restored = TeamsLedgerStateStore(state_path).load_projection(
        source_id="src-teams",
        conversation_id="19:conversation@thread.tacv2",
    )
    updated = projector.project_unthreaded(
        [
            _message("m1", "2026-07-08T09:30:00", "late earlier"),
            _message("m2", "2026-07-08T10:00:00", "anchor"),
            _message("m3", "2026-07-08T10:30:00", "follow-up"),
        ],
        previous=restored,
    )

    assert len(updated.blocks) == 1
    assert updated.blocks[0].window_id == original_block.window_id
    assert updated.blocks[0].frozen_anchor_message_id == "m2"
    assert updated.blocks[0].member_message_ids == ("m1", "m2", "m3")


def test_teams_ledger_state_store_persists_message_receipts_across_restart(tmp_path):
    state_path = tmp_path / "teams-ledger.json"
    first = TeamsLedgerStateStore(state_path).observe_messages(
        source_id="src-teams",
        conversation_id="19:conversation@thread.tacv2",
        messages=[
            _message("m1", "2026-07-08T09:00:00", "same"),
            _message("m2", "2026-07-08T09:01:00", "old"),
        ],
    )

    second = TeamsLedgerStateStore(state_path).observe_messages(
        source_id="src-teams",
        conversation_id="19:conversation@thread.tacv2",
        messages=[
            _message("m1", "2026-07-08T09:00:00", "same"),
            _message("m2", "2026-07-08T09:01:00", "new"),
            _message("m3", "2026-07-08T09:02:00", "fresh"),
        ],
    )

    assert first == {"new": 2, "updated": 0, "unchanged": 0}
    assert second == {"new": 1, "updated": 1, "unchanged": 1}
