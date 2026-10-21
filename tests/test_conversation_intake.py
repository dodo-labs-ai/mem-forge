"""Receiving a device collection: the request it answers, its manifest and bodies, and pruning once processed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memforge.conversations.collections import CollectedBody, StoredCollections
from memforge.conversations.intake import CollectionIntake, CollectionRejected
from memforge.conversations.layer import ConversationLayer, ConversationSelection
from memforge.conversations.manifest import (
    CollectionManifest,
    ConversationEvidence,
    ManifestMessage,
    PollEvidence,
    manifest_to_payload,
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
from memforge.conversations.reader import SourceProjectionUnitReader
from memforge.conversations.records import MessageIdentity, record_revision
from memforge.storage.database import Database

SOURCE = "src-chat"
ATTEMPT = "job-1:attempt:1"
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
SELECTION = ConversationSelection(conversation_ids=("c1", "c2"), initial_history_days=30)


@pytest.fixture
async def database(tmp_path):
    db = Database(str(tmp_path / "intake.db"))
    await db.connect()
    await db.upsert_source(
        id=SOURCE, type="teams", name="Teams", config_json="{}", access_policy="workspace", owner_user_id="dev"
    )
    try:
        yield db
    finally:
        await db.close()


def _collections(db: Database) -> StoredCollections:
    return StoredCollections(db, workspace_id="default")


def _intake(db: Database) -> CollectionIntake:
    return CollectionIntake(reader=SourceProjectionUnitReader(db), collections=_collections(db))


def _record(message_id: str, at: datetime, text: str = "hello") -> dict[str, object]:
    return {"id": message_id, "content": text, "time": at.isoformat(), "conversation_id": "c1"}


def _message(message_id: str, at: datetime) -> ManifestMessage:
    return ManifestMessage(
        identity=MessageIdentity("c1", message_id), revision=record_revision(_record(message_id, at)), created_at=at
    )


def _readable(conversation_id: str, *intervals: Interval) -> ConversationEvidence:
    return ConversationEvidence(
        conversation_id=conversation_id,
        status=ConversationStatus(ConversationAccess.READABLE, "listed"),
        ref=ConversationRef(
            label=ConversationLabel(conversation_id=conversation_id, kind="group_chat", name=conversation_id),
            threading=ThreadingModel.INLINE,
            poll_includes_replies=True,
        ),
        polls=tuple(
            PollEvidence(requested=interval, outcome=PollOutcome.REACHED_START, started_at=NOW)
            for interval in intervals
        ),
    )


def _payload(
    *conversations: ConversationEvidence, messages: tuple[ManifestMessage, ...] = (), attempt: str = ATTEMPT
) -> dict:
    payload = manifest_to_payload(
        CollectionManifest(
            source_id=SOURCE,
            collection_attempt_id=attempt,
            config_revision="cfg-1",
            record_schema="teams-message:2",
            received_at=NOW,
            conversations=conversations,
            messages=messages,
        )
    )
    del payload["received_at"]
    return payload


async def _issue(db: Database, attempt: str = ATTEMPT, *, now: datetime = NOW):
    request = ConversationLayer.collection_request(selection=SELECTION, now=now)
    return await _collections(db).issue_request(SOURCE, attempt, request, issued_at=now)


def _answer(request, *, messages: tuple[ManifestMessage, ...] = (), attempt: str = ATTEMPT) -> dict:
    return _payload(
        *(_readable(conversation.conversation_id, *conversation.intervals) for conversation in request.conversations),
        messages=messages,
        attempt=attempt,
    )


async def _receive(db: Database, payload: dict, *, received_at: datetime = NOW, attempt: str = ATTEMPT):
    return await _intake(db).receive_manifest(
        payload,
        source_id=SOURCE,
        collection_attempt_id=attempt,
        config_revision="cfg-1",
        record_schema="teams-message:2",
        received_at=received_at,
    )


async def test_an_attempt_keeps_the_first_request_issued_to_it(database):
    first = await _issue(database)
    again = await _issue(database, now=NOW + timedelta(minutes=5))

    assert again == first
    assert await _collections(database).request(SOURCE, ATTEMPT) == first


async def test_a_manifest_needs_the_request_of_its_attempt(database):
    with pytest.raises(CollectionRejected, match="was issued no request"):
        await _receive(database, _payload(_readable("c1", Interval(start=NOW - timedelta(days=30)))))


async def test_a_manifest_must_answer_every_requested_conversation_and_no_other(database):
    request = await _issue(database)
    (c1, _c2) = request.conversations

    with pytest.raises(CollectionRejected, match="exactly the requested conversations"):
        await _receive(database, _payload(_readable("c1", *c1.intervals)))


async def test_a_readable_conversation_must_be_polled_over_exactly_the_requested_intervals(database):
    request = await _issue(database)
    (c1, c2) = request.conversations
    wider = Interval(start=c1.intervals[0].start - timedelta(days=60))

    with pytest.raises(CollectionRejected, match="requested intervals"):
        await _receive(database, _payload(_readable("c1", wider), _readable("c2", *c2.intervals)))


async def test_an_attempt_keeps_its_first_manifest_and_refuses_another(database):
    request = await _issue(database)
    message = _message("m1", NOW - timedelta(hours=1))
    payload = _answer(request, messages=(message,))

    first = await _receive(database, payload)
    repeated = await _receive(database, payload, received_at=NOW + timedelta(minutes=1))
    with pytest.raises(CollectionRejected, match="another manifest"):
        await _receive(database, _answer(request))
    stored = await _collections(database).manifest(SOURCE, ATTEMPT)

    assert first == repeated
    assert [(item.identity.message_id, item.body) for item in first] == [("m1", True)]
    assert stored.received_at == NOW


async def test_a_body_must_match_its_manifest_message(database):
    request = await _issue(database)
    at = NOW - timedelta(hours=1)
    await _receive(database, _answer(request, messages=(_message("m1", at),)))
    intake = _intake(database)

    async def receive(record):
        await intake.receive_bodies(
            {MessageIdentity("c1", "m1"): CollectedBody(record=record)}, source_id=SOURCE, collection_attempt_id=ATTEMPT
        )

    with pytest.raises(CollectionRejected, match="manifest revision"):
        await receive(_record("m1", at, "tampered"))
    with pytest.raises(CollectionRejected, match="not in the attempt's manifest"):
        await intake.receive_bodies(
            {MessageIdentity("c1", "m9"): CollectedBody(record=_record("m9", at))},
            source_id=SOURCE,
            collection_attempt_id=ATTEMPT,
        )
    await receive(_record("m1", at))

    assert await intake.missing_bodies_of(SOURCE, ATTEMPT) == ()


async def test_processing_an_attempt_drops_its_bodies_and_everything_older_but_keeps_its_manifest(database):
    collections = _collections(database)
    at = NOW - timedelta(hours=1)
    attempts = ("job-0:attempt:1", ATTEMPT, "job-2:attempt:1")
    for offset, attempt in enumerate(attempts):
        moment = NOW + timedelta(minutes=10 * offset)
        request = await _issue(database, attempt, now=moment)
        await _receive(
            database,
            _answer(request, messages=(_message("m1", at),), attempt=attempt),
            received_at=moment,
            attempt=attempt,
        )
        await _intake(database).receive_bodies(
            {MessageIdentity("c1", "m1"): CollectedBody(record=_record("m1", at))},
            source_id=SOURCE,
            collection_attempt_id=attempt,
        )

    await collections.prune(SOURCE, ATTEMPT)

    assert await collections.manifest(SOURCE, "job-0:attempt:1") is None
    assert await collections.request(SOURCE, "job-0:attempt:1") is None
    assert await collections.manifest(SOURCE, ATTEMPT) is not None
    assert await collections.bodies(SOURCE, ATTEMPT) == {}
    assert await collections.request(SOURCE, "job-2:attempt:1") is not None
    assert len(await collections.bodies(SOURCE, "job-2:attempt:1")) == 1
    assert (await collections.latest_manifest(SOURCE)).collection_attempt_id == "job-2:attempt:1"


async def test_a_conversation_the_source_cannot_read_keeps_the_label_its_request_carried(database):
    request = await _issue(database)
    (c1, c2) = request.conversations
    refused = ConversationStatus(ConversationAccess.NO_ACCESS, "HTTP 403")
    relabelled = ConversationEvidence(
        conversation_id="c2",
        status=refused,
        last_known=ConversationLabel(conversation_id="c2", kind="group_chat", name="Someone else"),
    )

    with pytest.raises(CollectionRejected, match="does not keep the label"):
        await _receive(database, _payload(_readable("c1", *c1.intervals), relabelled))
    await _receive(
        database,
        _payload(
            _readable("c1", *c1.intervals),
            ConversationEvidence(conversation_id="c2", status=refused, last_known=c2.last_known),
        ),
    )


async def test_a_manifest_in_another_record_schema_is_refused(database):
    request = await _issue(database)

    with pytest.raises(CollectionRejected, match="not teams-message:3"):
        await _intake(database).receive_manifest(
            _answer(request),
            source_id=SOURCE,
            collection_attempt_id=ATTEMPT,
            config_revision="cfg-1",
            record_schema="teams-message:3",
            received_at=NOW,
        )


async def test_a_manifest_another_upload_recorded_first_is_refused(database, monkeypatch):
    request = await _issue(database)
    await _receive(database, _answer(request, messages=(_message("m1", NOW - timedelta(hours=1)),)))
    intake = _intake(database)
    collections = intake._collections

    async def read_before_the_other_upload(source_id, collection_attempt_id):
        return None

    monkeypatch.setattr(collections, "manifest", read_before_the_other_upload)

    with pytest.raises(CollectionRejected, match="another manifest"):
        await intake.receive_manifest(
            _answer(request),
            source_id=SOURCE,
            collection_attempt_id=ATTEMPT,
            config_revision="cfg-1",
            record_schema="teams-message:2",
            received_at=NOW,
        )
