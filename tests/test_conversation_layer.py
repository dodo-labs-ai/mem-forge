from __future__ import annotations

from collections.abc import Collection
from datetime import datetime, timedelta, timezone

import pytest

from memforge.conversations.coverage import JUST_AFTER
from memforge.conversations.layer import ConversationLayer, ConversationSelection
from memforge.conversations.manifest import (
    CollectionManifest,
    ConversationEvidence,
    ManifestMessage,
    PollEvidence,
    manifest_from_payload,
    manifest_to_payload,
    request_from_payload,
    request_to_payload,
)
from memforge.conversations.port import (
    ConversationAccess,
    ConversationLabel,
    ConversationRef,
    ConversationStatus,
    Interval,
    PollOutcome,
    ThreadingModel,
)
from memforge.conversations.records import MessageIdentity, record_revision
from memforge.conversations.units import Member, Span, StoredUnit, UnitKind, UnitRef
from memforge.source_projection import UnitRemovalReason

SOURCE = "src-chat"
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


class InMemoryProjection:
    """The current Units of one Source, answering the reads the conversation layer makes.

    A Unit's recorded span is its members' span, and every span bound is
    inclusive, as the Source Projection records them.
    """

    def __init__(self, *units: StoredUnit) -> None:
        self.units = list(units)

    def _of(self, conversation_id: str) -> list[StoredUnit]:
        return [unit for unit in self.units if unit.ref.conversation_id == conversation_id]

    async def holders(self, source_id: str, conversation_id: str, message_ids: Collection[str]):
        holders: dict[str, tuple[UnitRef, ...]] = {}
        for unit in self._of(conversation_id):
            for member in unit.members:
                if member.identity.message_id in message_ids:
                    holders[member.identity.message_id] = (*holders.get(member.identity.message_id, ()), unit.ref)
        return holders

    async def thread_units(self, source_id: str, conversation_id: str):
        return tuple(unit.ref for unit in self._of(conversation_id) if unit.kind is UnitKind.THREAD)

    async def read(self, units: Collection[UnitRef]):
        wanted = {unit.unit_id for unit in units}
        return tuple(unit for unit in self.units if unit.ref.unit_id in wanted)

    async def units_spanning(self, source_id: str, conversation_id: str, start: datetime, end: datetime):
        return tuple(
            unit
            for unit in self._of(conversation_id)
            if (span := Span.of(unit.members)) is not None and span.earliest <= end and span.latest >= start
        )

    async def units_ending_before(self, source_id: str, conversation_id: str, cutoff: datetime):
        return tuple(
            unit
            for unit in self._of(conversation_id)
            if (span := Span.of(unit.members)) is not None and span.latest < cutoff
        )

    async def units_of(self, source_id: str, conversation_id: str):
        return tuple(unit.ref for unit in self._of(conversation_id))

    async def conversations_held(self, source_id: str):
        return frozenset(unit.ref.conversation_id for unit in self.units)


def _record(conversation_id: str, message_id: str, at: datetime, text: str = "hello") -> dict[str, object]:
    return {"id": message_id, "content": text, "time": at.isoformat(), "conversation_id": conversation_id}


def _member(conversation_id: str, message_id: str, at: datetime) -> Member:
    record = _record(conversation_id, message_id, at)
    return Member(MessageIdentity(conversation_id, message_id), at, record, record_revision(record))


def _window(key: str, conversation_id: str, *members: Member) -> StoredUnit:
    return StoredUnit(
        ref=UnitRef(unit_id=key, document_id=f"doc-{key}", provider_key=key, kind=UnitKind.WINDOW,
                    conversation_id=conversation_id),
        members=members,
    )


def _ref(conversation_id: str) -> ConversationRef:
    return ConversationRef(
        label=ConversationLabel(conversation_id=conversation_id, kind="group_chat", name=conversation_id),
        threading=ThreadingModel.INLINE,
        poll_includes_replies=True,
    )


def _readable(conversation_id: str, *, outcome: PollOutcome = PollOutcome.REACHED_START, days: int = 30):
    return ConversationEvidence(
        conversation_id=conversation_id,
        status=ConversationStatus(ConversationAccess.READABLE, "ok"),
        ref=_ref(conversation_id),
        polls=(PollEvidence(requested=Interval(start=NOW - timedelta(days=days)), outcome=outcome, started_at=NOW),),
    )


def _status(conversation_id: str, access: ConversationAccess) -> ConversationEvidence:
    return ConversationEvidence(conversation_id=conversation_id, status=ConversationStatus(access, "provider said so"))


def _manifest(*conversations: ConversationEvidence, messages=(), attempt: str = "run-1:attempt:1"):
    return CollectionManifest(
        source_id=SOURCE,
        collection_attempt_id=attempt,
        config_revision="cfg-1",
        record_schema="conversation-message:1",
        received_at=NOW,
        conversations=conversations,
        messages=tuple(messages),
    )


def _manifest_message(conversation_id: str, message_id: str, at: datetime, text: str = "hello"):
    record = _record(conversation_id, message_id, at, text)
    message = ManifestMessage(
        identity=MessageIdentity(conversation_id, message_id), revision=record_revision(record), created_at=at
    )
    return message, record


def _selection(*conversation_ids: str, retention: int | None = None, history: int = 30) -> ConversationSelection:
    return ConversationSelection(
        conversation_ids=conversation_ids, initial_history_days=history, rolling_retention_days=retention
    )


def _removals(plan) -> list[tuple[str, UnitRemovalReason]]:
    return sorted((removal.unit.provider_key, removal.reason) for removal in plan.removals)


async def test_a_new_message_opens_a_window():
    message, record = _manifest_message("c1", "m1", NOW - timedelta(hours=1))
    layer = ConversationLayer(InMemoryProjection())

    plan = await layer.plan(
        manifest=_manifest(_readable("c1"), messages=[message]),
        bodies={message.identity: record},
        selection=_selection("c1"),
        attempt_started_at=NOW,
    )

    (unit,) = plan.units
    assert unit.kind is UnitKind.WINDOW
    assert [member.identity.message_id for member in unit.members] == ["m1"]
    assert plan.removals == () and plan.unreadable == ()


async def test_a_body_must_match_its_manifest_revision():
    message, record = _manifest_message("c1", "m1", NOW - timedelta(hours=1))
    layer = ConversationLayer(InMemoryProjection())

    with pytest.raises(ValueError, match="does not match"):
        await layer.plan(
            manifest=_manifest(_readable("c1"), messages=[message]),
            bodies={message.identity: {**record, "content": "tampered"}},
            selection=_selection("c1"),
            attempt_started_at=NOW,
        )


async def test_a_window_not_returned_inside_the_proven_interval_is_absent():
    old = _window("w-old", "c1", _member("c1", "m0", NOW - timedelta(days=2)))
    layer = ConversationLayer(InMemoryProjection(old))

    plan = await layer.plan(manifest=_manifest(_readable("c1")), bodies={}, selection=_selection("c1"),
                            attempt_started_at=NOW)

    assert _removals(plan) == [("w-old", UnitRemovalReason.ABSENT_FROM_SOURCE)]


async def test_running_out_of_history_removes_nothing_older_than_what_was_returned():
    old = _window("w-old", "c1", _member("c1", "m0", NOW - timedelta(days=20)))
    message, record = _manifest_message("c1", "m1", NOW - timedelta(days=1))
    evidence = ConversationEvidence(
        conversation_id="c1",
        status=ConversationStatus(ConversationAccess.READABLE, "ok"),
        ref=_ref("c1"),
        polls=(
            PollEvidence(
                requested=Interval(start=NOW - timedelta(days=30)),
                outcome=PollOutcome.HISTORY_EXHAUSTED,
                started_at=NOW,
                earliest_returned_at=message.created_at,
            ),
        ),
    )
    layer = ConversationLayer(InMemoryProjection(old))

    plan = await layer.plan(manifest=_manifest(evidence, messages=[message]), bodies={message.identity: record},
                            selection=_selection("c1"), attempt_started_at=NOW)

    assert plan.removals == ()


async def test_a_gone_conversation_loses_every_unit():
    projection = InMemoryProjection(
        _window("w1", "c1", _member("c1", "m1", NOW - timedelta(days=200))),
        _window("w2", "c1", _member("c1", "m2", NOW - timedelta(days=1))),
    )
    layer = ConversationLayer(projection)

    plan = await layer.plan(manifest=_manifest(_status("c1", ConversationAccess.GONE)), bodies={},
                            selection=_selection("c1"), attempt_started_at=NOW)

    assert _removals(plan) == [("w1", UnitRemovalReason.ABSENT_FROM_SOURCE), ("w2", UnitRemovalReason.ABSENT_FROM_SOURCE)]


async def test_lost_access_keeps_support_but_still_applies_retention_and_is_reported():
    projection = InMemoryProjection(
        _window("w-old", "c1", _member("c1", "m1", NOW - timedelta(days=400))),
        _window("w-new", "c1", _member("c1", "m2", NOW - timedelta(days=1))),
        _window("w-other", "c2", _member("c2", "m3", NOW - timedelta(days=1))),
    )
    layer = ConversationLayer(projection)

    plan = await layer.plan(
        manifest=_manifest(_status("c1", ConversationAccess.NO_ACCESS), _readable("c2", days=365)),
        bodies={},
        selection=_selection("c1", "c2", retention=365, history=365),
        attempt_started_at=NOW,
    )

    assert _removals(plan) == [("w-old", UnitRemovalReason.OUTSIDE_ROLLING_RETENTION), ("w-other", UnitRemovalReason.ABSENT_FROM_SOURCE)]
    assert [evidence.conversation_id for evidence in plan.unreadable] == ["c1"]


async def test_a_conversation_without_a_status_in_this_attempt_is_left_alone():
    projection = InMemoryProjection(_window("w-old", "c1", _member("c1", "m1", NOW - timedelta(days=400))))
    layer = ConversationLayer(projection)

    plan = await layer.plan(manifest=_manifest(), bodies={}, selection=_selection("c1", retention=365),
                            attempt_started_at=NOW)

    assert plan.removals == ()


async def test_deselected_conversations_are_removed_from_scope():
    projection = InMemoryProjection(_window("w", "c-gone", _member("c-gone", "m", NOW - timedelta(days=1))))
    layer = ConversationLayer(projection)

    plan = await layer.plan(manifest=_manifest(), bodies={}, selection=_selection("c1"), attempt_started_at=NOW,
                            deselected_conversation_ids=["c-gone"])

    assert _removals(plan) == [("w", UnitRemovalReason.REMOVED_FROM_SCOPE)]


async def test_a_manifest_conversation_outside_the_selection_is_refused():
    layer = ConversationLayer(InMemoryProjection())

    with pytest.raises(ValueError, match="outside the selection"):
        await layer.plan(manifest=_manifest(_readable("c9")), bodies={}, selection=_selection("c1"),
                         attempt_started_at=NOW)


def test_initial_history_cannot_exceed_rolling_retention():
    with pytest.raises(ValueError, match="cannot exceed"):
        _selection("c1", retention=30, history=60)


def test_the_request_polls_initial_history_within_the_ceiling_and_retention():
    selection = ConversationSelection(
        conversation_ids=("c1",), initial_history_days=60, rolling_retention_days=365,
        regular_ceiling=timedelta(days=30),
    )

    request = ConversationLayer.collection_request(selection=selection, now=NOW)

    (conversation,) = request.conversations
    assert conversation.intervals == (Interval(start=NOW - timedelta(days=30)),)
    assert request_from_payload(request_to_payload(request)) == request


def test_a_poll_stopped_early_is_backfilled_by_the_next_attempt():
    stopped_at = NOW - timedelta(days=10)
    previous = _manifest(
        ConversationEvidence(
            conversation_id="c1",
            status=ConversationStatus(ConversationAccess.READABLE, "ok"),
            ref=_ref("c1"),
            polls=(
                PollEvidence(
                    requested=Interval(start=NOW - timedelta(days=30)),
                    outcome=PollOutcome.STOPPED_EARLY,
                    started_at=NOW - timedelta(hours=1),
                    earliest_complete_page_time=stopped_at,
                ),
            ),
        )
    )
    request = ConversationLayer.collection_request(selection=_selection("c1"), now=NOW, previous=previous)

    (conversation,) = request.conversations
    assert conversation.intervals == (
        Interval(start=NOW - timedelta(days=30)),
        Interval(start=NOW - timedelta(days=30), end=stopped_at + JUST_AFTER),
    )
    assert conversation.last_known == _ref("c1").label


def test_the_manifest_round_trips_through_its_payload():
    message, _record_value = _manifest_message("c1", "m1", NOW - timedelta(hours=1))
    manifest = _manifest(_readable("c1"), _status("c2", ConversationAccess.NO_ACCESS), messages=[message])

    assert manifest_from_payload(manifest_to_payload(manifest)) == manifest


def test_a_manifest_message_must_belong_to_a_readable_conversation():
    message, _record_value = _manifest_message("c2", "m1", NOW)

    with pytest.raises(ValueError, match="readable conversation"):
        _manifest(_status("c2", ConversationAccess.NO_ACCESS), messages=[message])


async def test_a_new_unit_ending_before_the_retention_cutoff_is_not_projected():
    message, record = _manifest_message("c1", "m-old", NOW - timedelta(days=40))
    layer = ConversationLayer(InMemoryProjection())

    plan = await layer.plan(
        manifest=_manifest(_readable("c1", days=60), messages=[message]),
        bodies={message.identity: record},
        selection=_selection("c1", retention=30, history=30),
        attempt_started_at=NOW,
    )

    assert plan.units == () and plan.removals == ()


async def test_a_changed_unit_ending_before_the_retention_cutoff_is_removed():
    old = _member("c1", "m-old", NOW - timedelta(days=40))
    layer = ConversationLayer(InMemoryProjection(_window("w-old", "c1", old)))
    message, _ = _manifest_message("c1", "m-old", old.created_at, "edited")
    record = _record("c1", "m-old", old.created_at, "edited")

    plan = await layer.plan(
        manifest=_manifest(_readable("c1", days=60), messages=[message]),
        bodies={message.identity: record},
        selection=_selection("c1", retention=30, history=30),
        attempt_started_at=NOW,
    )

    assert plan.units == ()
    assert _removals(plan) == [("w-old", UnitRemovalReason.OUTSIDE_ROLLING_RETENTION)]


async def test_a_held_message_needs_no_body_at_its_stored_revision():
    held = _member("c1", "m1", NOW - timedelta(hours=2))
    layer = ConversationLayer(InMemoryProjection(_window("w", "c1", held)))
    message, _ = _manifest_message("c1", "m1", held.created_at)
    new, new_record = _manifest_message("c1", "m2", NOW - timedelta(hours=1))

    plan = await layer.plan(
        manifest=_manifest(_readable("c1"), messages=[message, new]),
        bodies={new.identity: new_record},
        selection=_selection("c1"),
        attempt_started_at=NOW,
    )

    (unit,) = plan.units
    assert unit.provider_key == "w"
    assert [member.identity.message_id for member in unit.members] == ["m1", "m2"]


async def test_a_message_whose_waived_body_is_no_longer_held_is_kept_until_the_next_attempt():
    held = _member("c1", "m1", NOW - timedelta(hours=2))
    layer = ConversationLayer(InMemoryProjection(_window("w", "c1", held)))
    edited, _ = _manifest_message("c1", "m1", held.created_at, "edited since the body was waived")

    plan = await layer.plan(
        manifest=_manifest(_readable("c1"), messages=[edited]),
        bodies={},
        selection=_selection("c1"),
        attempt_started_at=NOW,
    )

    assert plan.units == () and plan.removals == ()
