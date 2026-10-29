"""A Unit that holds members stored without their own creation time, as payload, text and Observations."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from memforge.conversations.layer import _returned
from memforge.conversations.manifest import ManifestMessage
from memforge.conversations.membership import PlannedUnit
from memforge.conversations.port import ConversationLabel
from memforge.conversations.projection import normalize_unit, unit_content_item, unit_payload, unit_raw_content
from memforge.conversations.records import MessageIdentity, record_revision
from memforge.conversations.undated import UNDATED_MEMBER_TIMES_KEY
from memforge.conversations.units import Member, UnitKind
from memforge.source_adapters.conversation import project_native
from memforge.source_time import SOURCE_UPDATED_AT_KEY

CHAT = "c1"
RECORD_SCHEMA = "teams-message:2"
LABEL = ConversationLabel(conversation_id=CHAT, kind="group_chat", name="Release crew")


def _at(hour: int) -> datetime:
    return datetime(2026, 9, 1, hour, tzinfo=timezone.utc)


def _dated(message_id: str, hour: int) -> Member:
    record = {"content": f"dated {message_id}", "from": "Ada", "time": _at(hour).isoformat(), "conversation_id": CHAT}
    return Member(MessageIdentity(CHAT, message_id), _at(hour), record, record_revision(record), RECORD_SCHEMA)


def _undated(message_id: str, hour: int) -> Member:
    record = {"attachments": [], "content": f"undated {message_id}", "deleted": False}
    return Member(MessageIdentity(CHAT, message_id), _at(hour), record, record_revision(record), RECORD_SCHEMA)


def _projected(*members: Member):
    unit = PlannedUnit(UnitKind.WINDOW, "window-key", LABEL, members)
    payload = unit_payload(unit, record_schema=RECORD_SCHEMA)
    item = unit_content_item(unit, source_id="src")
    return payload, item, normalize_unit(unit_raw_content(item, payload, with_artifacts=False))


def test_a_unit_of_undated_members_is_placed_and_shown_by_their_stored_times():
    payload, item, normalized = _projected(_undated("a", 9), _undated("b", 10))

    assert payload[UNDATED_MEMBER_TIMES_KEY] == {"a": _at(9).isoformat(), "b": _at(10).isoformat()}
    assert item.last_modified == _at(10)
    assert "**Date range**: 2026-09-01T09:00 – 2026-09-01T10:00" in normalized.markdown_body
    assert "**Unknown** (2026-09-01T09:00):" in normalized.markdown_body
    assert normalized.source_semantics[SOURCE_UPDATED_AT_KEY] == _at(10).isoformat()

    projection = project_native(item=item, native=payload)

    assert [observation.observed_at for observation in projection.observations] == [
        _at(9).isoformat(),
        _at(10).isoformat(),
    ]
    assert (projection.locator["observed_from"], projection.locator["observed_to"]) == (
        _at(9).isoformat(),
        _at(10).isoformat(),
    )


def test_a_mixed_unit_names_only_its_undated_members_and_shows_every_time():
    payload, _item, normalized = _projected(_undated("a", 9), _dated("b", 10))

    assert payload[UNDATED_MEMBER_TIMES_KEY] == {"a": _at(9).isoformat()}
    assert "**Date range**: 2026-09-01T09:00 – 2026-09-01T10:00" in normalized.markdown_body
    assert "**Ada** (2026-09-01T10:00):" in normalized.markdown_body


def test_a_unit_of_dated_members_carries_no_undated_times():
    payload, _item, _normalized = _projected(_dated("a", 9))

    assert UNDATED_MEMBER_TIMES_KEY not in payload


def test_whether_a_member_is_undated_is_derived_from_its_record_when_it_is_built():
    dated, undated = _dated("a", 9), _undated("b", 10)

    assert not dated.undated
    assert undated.undated
    assert replace(dated, record=undated.record).undated
    assert not replace(undated, record=dated.record).undated
    assert dated == replace(dated)

    # A message returned at the revision a Unit holds reuses the stored undated record.
    message = ManifestMessage(identity=undated.identity, revision=undated.revision, created_at=undated.created_at)
    reused = _returned(message, {}, {}, {undated.identity: undated}, record_schema=RECORD_SCHEMA)

    assert reused is not None and reused.member.record is undated.record
    assert reused.member.undated
