"""Receiving a device collection: the request it answers, its manifest and bodies, and pruning once processed."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from memforge.conversations.collections import CollectedBody, StoredCollections
from memforge.conversations.intake import AgentUpgradeRequired, CollectionIntake, CollectionRejected
from memforge.conversations.layer import ConversationLayer, ConversationSelection
from memforge.conversations.manifest import (
    CollectionManifest,
    CollectionRequest,
    ConversationEvidence,
    ManifestMessage,
    PollEvidence,
    RequestedConversation,
    manifest_to_payload,
    predates_record_schema,
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
            PollEvidence(
                requested=interval,
                outcome=PollOutcome.REACHED_START,
                started_at=NOW,
                earliest_complete_page_time=interval.start - timedelta(minutes=1),
            )
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


def _with_polls(evidence: ConversationEvidence, **changes) -> ConversationEvidence:
    return replace(evidence, polls=tuple(replace(poll, **changes) for poll in evidence.polls))


@pytest.mark.parametrize(
    "complete_from",
    [None, "at_start"],
    ids=["without a complete time", "complete only from the start itself"],
)
async def test_a_poll_that_reached_its_start_must_be_complete_from_before_it(database, complete_from):
    request = await _issue(database)
    (c1, c2) = request.conversations
    start = c1.intervals[0].start
    claimed = _with_polls(
        _readable("c1", *c1.intervals), earliest_complete_page_time=start if complete_from == "at_start" else None
    )

    with pytest.raises(CollectionRejected, match="reached its start without a complete time before it; upgrade"):
        await _receive(database, _payload(claimed, _readable("c2", *c2.intervals)))


@pytest.mark.parametrize("listed_at", ["unlisted", "before the interval"])
async def test_a_poll_names_as_its_earliest_message_only_one_it_could_have_returned(database, listed_at):
    request = await _issue(database)
    (c1, c2) = request.conversations
    before = c1.intervals[0].start - timedelta(days=1)
    listed = _message("m1", NOW - timedelta(hours=1) if listed_at == "unlisted" else before)
    claimed = _with_polls(
        _readable("c1", *c1.intervals),
        earliest_returned_at=NOW - timedelta(hours=3) if listed_at == "unlisted" else before,
    )

    with pytest.raises(CollectionRejected, match="earliest message the manifest does not list inside its interval"):
        await _receive(database, _payload(claimed, _readable("c2", *c2.intervals), messages=(listed,)))


NOT_REACHING_START = (PollOutcome.HISTORY_EXHAUSTED, PollOutcome.STOPPED_EARLY)


def _outcome_id(outcome: PollOutcome) -> str:
    return outcome.value


async def _issue_with_backfill(db: Database) -> tuple[Interval, Interval]:
    """Issue a request whose c1 carries a regular poll and a backfill below it, and return both intervals."""

    regular = Interval(start=NOW - timedelta(days=10))
    backfill = Interval(start=NOW - timedelta(days=30), end=NOW - timedelta(days=10))
    request = CollectionRequest(
        conversations=(
            RequestedConversation(conversation_id="c1", intervals=(regular, backfill)),
            RequestedConversation(conversation_id="c2", intervals=(regular,)),
        )
    )
    await _collections(db).issue_request(SOURCE, ATTEMPT, request, issued_at=NOW)
    return regular, backfill


def _polled(
    regular: Interval, backfill: Interval, *, outcome: PollOutcome, complete_from: datetime | None, returned: bool
) -> tuple[dict, ManifestMessage | None]:
    """A manifest whose c1 backfill poll ended with ``outcome`` and names ``complete_from``."""

    message = _message("m1", backfill.start + timedelta(days=1)) if returned else None
    c1 = _readable("c1", regular, backfill)
    c1 = replace(
        c1,
        polls=(
            c1.polls[0],
            replace(
                c1.polls[1],
                outcome=outcome,
                earliest_complete_page_time=complete_from,
                earliest_returned_at=message.created_at if message else None,
            ),
        ),
    )
    return _payload(c1, _readable("c2", regular), messages=(message,) if message else ()), message


@pytest.mark.parametrize("outcome", list(PollOutcome), ids=_outcome_id)
async def test_a_poll_that_returned_a_message_must_name_the_time_its_read_is_complete_from(database, outcome):
    regular, backfill = await _issue_with_backfill(database)
    payload, _ = _polled(regular, backfill, outcome=outcome, complete_from=None, returned=True)

    with pytest.raises(CollectionRejected, match="complete .*; upgrade the MemForge agent that collected it"):
        await _receive(database, payload)


@pytest.mark.parametrize("outcome", NOT_REACHING_START, ids=_outcome_id)
async def test_a_poll_that_ran_out_of_history_or_stopped_early_cannot_be_complete_from_before_its_start(
    database, outcome
):
    regular, backfill = await _issue_with_backfill(database)
    before = backfill.start - timedelta(milliseconds=1)
    payload, _ = _polled(regular, backfill, outcome=outcome, complete_from=before, returned=True)

    with pytest.raises(CollectionRejected, match="is complete from before its requested start; upgrade"):
        await _receive(database, payload)


@pytest.mark.parametrize("complete_from", ["at_end", "after_end"])
async def test_a_poll_stopped_early_must_be_complete_from_inside_its_interval(database, complete_from):
    regular, backfill = await _issue_with_backfill(database)
    late = backfill.end if complete_from == "at_end" else backfill.end + timedelta(hours=1)
    payload, _ = _polled(regular, backfill, outcome=PollOutcome.STOPPED_EARLY, complete_from=late, returned=True)

    with pytest.raises(CollectionRejected, match="is complete from at or after its requested end; upgrade"):
        await _receive(database, payload)


@pytest.mark.parametrize(
    "outcome, complete_from",
    [
        (PollOutcome.HISTORY_EXHAUSTED, "start"),
        (PollOutcome.STOPPED_EARLY, "start"),
        (PollOutcome.STOPPED_EARLY, "just_before_end"),
    ],
    ids=["history exhausted from its start", "stopped early from its start", "stopped early just before its end"],
)
async def test_a_poll_complete_from_inside_its_interval_is_accepted(database, outcome, complete_from):
    regular, backfill = await _issue_with_backfill(database)
    at = backfill.start if complete_from == "start" else backfill.end - timedelta(milliseconds=1)
    payload, message = _polled(regular, backfill, outcome=outcome, complete_from=at, returned=True)

    received = await _receive(database, payload)

    assert [item.identity for item in received] == [message.identity]


@pytest.mark.parametrize("outcome", NOT_REACHING_START, ids=_outcome_id)
async def test_a_poll_that_returned_nothing_may_name_no_complete_time(database, outcome):
    regular, backfill = await _issue_with_backfill(database)
    payload, _ = _polled(regular, backfill, outcome=outcome, complete_from=None, returned=False)

    assert await _receive(database, payload) == ()


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


@pytest.mark.parametrize(
    ("schema", "other", "earlier"),
    [
        ("teams-message:2", "teams-message:3", True),
        ("teams-message:3", "teams-message:10", True),
        ("teams-message:10", "teams-message:3", False),
        ("teams-message:3", "teams-message:3", False),
        ("teams-message:2", "jira-comment:3", False),
        ("teams-message:x", "teams-message:3", False),
        ("teams-message:\u00b2", "teams-message:3", False),
    ],
)
def test_a_record_schema_predates_only_a_later_version_of_itself(schema, other, earlier):
    assert predates_record_schema(schema, other) is earlier


@pytest.mark.parametrize(
    ("server_reads", "refusal", "reason"),
    [
        ("teams-message:3", AgentUpgradeRequired, "teams-message:2, an earlier version of the teams-message:3"),
        ("teams-message:1", CollectionRejected, "^server_upgrade_required: .*teams-message:2, a later version of the teams-message:1 this server reads$"),
        ("jira-comment:6", CollectionRejected, "teams-message:2, not jira-comment:6"),
    ],
    ids=["agent older", "server older", "other schema"],
)
async def test_a_manifest_in_another_record_schema_is_refused(database, server_reads, refusal, reason):
    request = await _issue(database)

    with pytest.raises(refusal, match=reason) as refused:
        await _intake(database).receive_manifest(
            _answer(request),
            source_id=SOURCE,
            collection_attempt_id=ATTEMPT,
            config_revision="cfg-1",
            record_schema=server_reads,
            received_at=NOW,
        )
    assert isinstance(refused.value, AgentUpgradeRequired) is (refusal is AgentUpgradeRequired)


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
