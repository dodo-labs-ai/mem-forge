"""The facts every registered conversation provider owes the conversation layer.

The layer decides what a collection proves and what it removes; a provider
only reports. This suite checks those reports against a fake of each
provider's service, whose messages are the ground truth: statuses, the half-
open interval a poll returns, the paging evidence a poll gives, and what the
records guarantee. It is parameterized by the registered conversation source
types, so a new provider joins it by registering and fails it until
``tests/contracts/conversation_fakes`` has a module for it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta

import pytest

from memforge.conversations.coverage import proven_interval
from memforge.conversations.intake import _require_consistent_poll
from memforge.conversations.manifest import PollEvidence
from memforge.conversations.port import (
    ConversationAccess,
    ConversationPoll,
    ConversationProvider,
    ConversationRef,
    ConversationUnavailable,
    Interval,
    PollOutcome,
)
from memforge.conversations.records import record_created_at
from memforge.genes.teams_gene import TeamsGene
from memforge.source_types import source_type_descriptor
from tests.contracts.conversation_fakes import (
    NOW,
    ConversationProviderFakes,
    MissingConversationFakes,
    Scenario,
    ScenarioMessage,
    conversation_source_types,
    fakes_for,
    fakes_modules,
)
from tests.source_type_fixture import register_test_source_type

CONVERSATION_TYPES = conversation_source_types()

# Ten top-level messages, one an hour: m1 an hour before NOW, m10 ten hours before.
HOURLY = tuple(ScenarioMessage(f"m{hours}", NOW - timedelta(hours=hours)) for hours in range(1, 11))
BEFORE_EVERY_MESSAGE = NOW - timedelta(hours=30)


def _fakes_or_none(source_type: str) -> ConversationProviderFakes | None:
    try:
        return fakes_for(source_type)
    except MissingConversationFakes:
        return None


def _answer_cases(answers_of) -> list:
    """One case per source type with fakes and per answer ``answers_of(fakes)`` names."""

    cases = []
    for source_type in CONVERSATION_TYPES:
        fakes = _fakes_or_none(source_type)
        if fakes is None:
            continue
        cases.extend(pytest.param(source_type, answer, id=f"{source_type}-{answer}") for answer in answers_of(fakes))
    return cases


async def _ref(provider: ConversationProvider, conversation_id: str) -> ConversationRef:
    """The conversation's reference, from the listing or else read by id, as an executor finds it."""

    listing = await provider.list_conversations()
    listed = {ref.conversation_id: ref for ref in listing.conversations}
    if conversation_id in listed:
        return listed[conversation_id]
    _status, ref = await provider.get_conversation(conversation_id)
    assert ref is not None
    return ref


async def _poll(
    fakes: ConversationProviderFakes, scenario: Scenario, interval: Interval, *, stop_after: int | None = None
) -> tuple[ConversationRef, ConversationPoll]:
    async with fakes.connect(scenario) as connection:
        if stop_after is not None:
            connection.stop_after(stop_after)
        ref = await _ref(connection.provider, fakes.conversation_id)
        return ref, await connection.provider.poll(ref, interval)


def _evidence(interval: Interval, poll: ConversationPoll) -> PollEvidence:
    """The poll's evidence as an executor records it, for a poll started at NOW."""

    return PollEvidence(
        requested=interval,
        outcome=poll.outcome,
        started_at=NOW,
        earliest_returned_at=min((message.created_at for message in poll.messages), default=None),
        earliest_complete_page_time=poll.earliest_complete_page_time,
        stop_reason=poll.stop_reason,
    )


_MISSING = object()


def _at(record: Mapping[str, object], pointer: str) -> object:
    """The value at a JSON pointer of a record, or ``_MISSING``."""

    value: object = record
    for token in pointer.split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or key not in value:
            return _MISSING
        value = value[key]
    return value


# ---------------------------------------------------------------------------
# The suite covers every registered conversation type
# ---------------------------------------------------------------------------


def test_the_suite_covers_exactly_the_registered_conversation_types() -> None:
    assert CONVERSATION_TYPES == conversation_source_types()
    assert fakes_modules() == set(CONVERSATION_TYPES)
    assert all(isinstance(fakes_for(source_type), ConversationProviderFakes) for source_type in CONVERSATION_TYPES)


def test_a_registered_conversation_type_without_fakes_fails_the_suite(monkeypatch) -> None:
    register_test_source_type(
        monkeypatch,
        "fakeless_conversations",
        TeamsGene,
        conversation_provider=source_type_descriptor("teams").conversation_provider,
    )

    assert "fakeless_conversations" in conversation_source_types()
    with pytest.raises(MissingConversationFakes, match="tests/contracts/conversation_fakes/fakeless_conversations.py"):
        fakes_for("fakeless_conversations")


# ---------------------------------------------------------------------------
# Statuses and errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_type", CONVERSATION_TYPES)
async def test_gone_is_reported_only_for_an_answer_that_cannot_mean_lost_access(source_type) -> None:
    fakes = fakes_for(source_type)

    for answer in (*fakes.refused_answers, *fakes.failing_answers):
        reported: list[ConversationAccess] = []
        async with fakes.connect(Scenario(HOURLY), answer=answer) as connection:
            provider = connection.provider
            try:
                reported.extend(status.access for status in (await provider.list_conversations()).statuses.values())
            except Exception:
                pass
            ref = None
            try:
                status, ref = await provider.get_conversation(fakes.conversation_id)
                reported.append(status.access)
            except Exception:
                pass
            if ref is not None:
                try:
                    await provider.poll(ref, Interval(start=BEFORE_EVERY_MESSAGE))
                except ConversationUnavailable as unavailable:
                    reported.append(unavailable.status.access)
                except Exception:
                    pass
        assert ConversationAccess.GONE not in reported, answer

    for answer in fakes.gone_answers:
        async with fakes.connect(Scenario(HOURLY), answer=answer) as connection:
            status, _ref = await connection.provider.get_conversation(fakes.conversation_id)
        assert status.access is ConversationAccess.GONE, answer


@pytest.mark.parametrize(("source_type", "answer"), _answer_cases(lambda fakes: fakes.refused_answers))
async def test_a_refused_poll_raises_conversation_unavailable_with_no_access(source_type, answer) -> None:
    fakes = fakes_for(source_type)
    async with fakes.connect(Scenario(HOURLY)) as connection:
        ref = await _ref(connection.provider, fakes.conversation_id)
    async with fakes.connect(Scenario(HOURLY), answer=answer) as connection:
        with pytest.raises(ConversationUnavailable) as refused:
            await connection.provider.poll(ref, Interval(start=BEFORE_EVERY_MESSAGE))

    assert refused.value.status.access is ConversationAccess.NO_ACCESS


@pytest.mark.parametrize(("source_type", "answer"), _answer_cases(lambda fakes: fakes.failing_answers))
async def test_any_other_provider_error_raises(source_type, answer) -> None:
    fakes = fakes_for(source_type)
    async with fakes.connect(Scenario(HOURLY)) as connection:
        ref = await _ref(connection.provider, fakes.conversation_id)
    async with fakes.connect(Scenario(HOURLY), answer=answer) as connection:
        with pytest.raises(Exception) as failed:
            await connection.provider.poll(ref, Interval(start=BEFORE_EVERY_MESSAGE))

    # Raising, the poll returned no messages; and it must not read as a refused conversation.
    assert not isinstance(failed.value, ConversationUnavailable)


# ---------------------------------------------------------------------------
# What a poll returns and what its evidence proves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [2, 200])
@pytest.mark.parametrize("source_type", CONVERSATION_TYPES)
async def test_a_poll_returns_exactly_the_half_open_interval(source_type, page_size) -> None:
    fakes = fakes_for(source_type)
    step = fakes.time_precision
    start, end = NOW - timedelta(hours=6), NOW - timedelta(hours=2)
    scenario = Scenario(
        (
            ScenarioMessage("before-start", start - step),
            ScenarioMessage("at-start", start),
            ScenarioMessage("inside", NOW - timedelta(hours=4)),
            ScenarioMessage("before-end", end - step),
            ScenarioMessage("at-end", end),
            ScenarioMessage("after-end", NOW - timedelta(hours=1)),
        ),
        page_size=page_size,
    )

    for interval, expected in (
        (Interval(start, end), {"at-start", "inside", "before-end"}),
        (Interval(start), {"at-start", "inside", "before-end", "at-end", "after-end"}),
    ):
        _ref, poll = await _poll(fakes, scenario, interval)

        assert {message.message_id for message in poll.messages} == expected, interval


@dataclass(frozen=True)
class PagingShape:
    scenario: Scenario
    interval: Interval
    outcome: PollOutcome
    stop_after: int | None = None


PAGING_SHAPES = {
    "reached-start": PagingShape(
        Scenario(HOURLY, page_size=2), Interval(NOW - timedelta(hours=5, minutes=30)), PollOutcome.REACHED_START
    ),
    "history-exhausted-on-the-oldest-page": PagingShape(
        Scenario(HOURLY, page_size=2), Interval(BEFORE_EVERY_MESSAGE), PollOutcome.HISTORY_EXHAUSTED
    ),
    "history-exhausted-on-an-empty-page": PagingShape(
        Scenario(HOURLY, page_size=2, ends_on_empty_page=True),
        Interval(BEFORE_EVERY_MESSAGE),
        PollOutcome.HISTORY_EXHAUSTED,
    ),
    "stopped-early-after-one-page": PagingShape(
        Scenario(HOURLY, page_size=2), Interval(BEFORE_EVERY_MESSAGE), PollOutcome.STOPPED_EARLY, stop_after=1
    ),
    "stopped-early-after-three-pages": PagingShape(
        Scenario(HOURLY, page_size=2), Interval(BEFORE_EVERY_MESSAGE), PollOutcome.STOPPED_EARLY, stop_after=3
    ),
    # A message created long ago and deleted recently: a provider that places
    # it where it was deleted serves it on a newer page, and its time must not
    # extend what that page is complete from.
    "an-older-message-on-a-newer-page": PagingShape(
        Scenario(
            (
                *HOURLY,
                ScenarioMessage(
                    "deleted", NOW - timedelta(hours=8, minutes=30), deleted_at=NOW - timedelta(minutes=30)
                ),
            ),
            page_size=2,
        ),
        Interval(BEFORE_EVERY_MESSAGE),
        PollOutcome.STOPPED_EARLY,
        stop_after=2,
    ),
}


@pytest.mark.parametrize("shape", PAGING_SHAPES.values(), ids=PAGING_SHAPES.keys())
@pytest.mark.parametrize("source_type", CONVERSATION_TYPES)
async def test_paging_evidence_never_claims_more_than_the_pages_read(source_type, shape) -> None:
    fakes = fakes_for(source_type)

    ref, poll = await _poll(fakes, shape.scenario, shape.interval, stop_after=shape.stop_after)

    evidence = _evidence(shape.interval, poll)
    proven = proven_interval(evidence, ref, received_at=NOW, margin=timedelta(0))
    returned = {message.message_id for message in poll.messages}
    assert poll.outcome is shape.outcome
    assert not proven.empty
    assert returned <= shape.scenario.live_between(shape.interval.start, shape.interval.end)
    assert {
        message.message_id for message in shape.scenario.messages if message.live and proven.contains(message.created_at)
    } <= returned
    assert poll.earliest_complete_page_time is not None
    _require_consistent_poll(fakes.conversation_id, evidence, {message.created_at for message in poll.messages})


# ---------------------------------------------------------------------------
# Records and references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_type", CONVERSATION_TYPES)
async def test_records_match_the_declared_record_schema(source_type) -> None:
    fakes = fakes_for(source_type)
    scenario = Scenario(
        (
            ScenarioMessage("root", NOW - timedelta(hours=3)),
            ScenarioMessage("deleted", NOW - timedelta(hours=2, minutes=30), deleted_at=NOW - timedelta(hours=1)),
            ScenarioMessage("reply", NOW - timedelta(hours=2), reply_to="root"),
            ScenarioMessage("standalone", NOW - timedelta(hours=1, minutes=30)),
        )
    )
    truth = {message.message_id: message for message in scenario.messages}

    async with fakes.connect(scenario) as connection:
        schema = connection.provider.record_schema
        ref = await _ref(connection.provider, fakes.conversation_id)
        poll = await connection.provider.poll(ref, Interval(BEFORE_EVERY_MESSAGE))

    text_pointers = [field.json_pointer for field in schema.fields if field.nested_profile or field.text_format]
    assert {message.message_id for message in poll.messages} >= {"root", "reply", "standalone"}
    for message in poll.messages:
        record, expected = message.record, truth[message.message_id]
        assert record["representation"] == schema.qualified_name
        assert record_created_at(record) == message.created_at == expected.created_at
        assert _at(record, "/conversation_id") == fakes.conversation_id
        if expected.reply_to is None:
            assert _at(record, "/reply_to_id") in (None, _MISSING), message.message_id
        else:
            assert _at(record, "/reply_to_id") not in (None, _MISSING, ""), message.message_id
        if schema.tombstone_pointer is not None:
            tombstone = _at(record, schema.tombstone_pointer)
            assert (tombstone not in (None, _MISSING, "")) is (not expected.live), message.message_id
        for pointer in text_pointers:
            value = _at(record, pointer)
            assert value is _MISSING or value is None or isinstance(value, str), (message.message_id, pointer)


@pytest.mark.parametrize("source_type", CONVERSATION_TYPES)
async def test_a_provider_without_the_thread_capability_returns_only_references_whose_polls_include_replies(
    source_type,
) -> None:
    descriptor = source_type_descriptor(source_type)
    assert descriptor is not None and descriptor.conversation_provider is not None
    if getattr(descriptor.conversation_provider.provider_class(), "polls_omit_replies", False):
        pytest.skip(f"{source_type} fetches threads, so its polls may omit replies")
    fakes = fakes_for(source_type)

    async with fakes.connect(Scenario(HOURLY)) as connection:
        listed = (await connection.provider.list_conversations()).conversations
        _status, by_id = await connection.provider.get_conversation(fakes.conversation_id)

    refs = [*listed, *([by_id] if by_id is not None else [])]
    assert refs
    assert all(ref.poll_includes_replies for ref in refs)

