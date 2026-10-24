"""Teams as a conversation provider: its records, threads, statuses and conversation references."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from memforge.conversations.port import (
    ConversationAccess,
    Interval,
    PollOutcome,
    ThreadingModel,
)
from memforge.genes.teams_client import TeamsChatClient, TeamsMessagePoll
from memforge.genes.teams_provider import (
    TeamsConversationProvider,
    teams_conversation_ref,
    teams_message_record,
    teams_provider_message,
)
from memforge.source_adapters.teams import message_record

SENT = datetime(2026, 7, 14, 10, 0, tzinfo=timezone.utc)
EDITED = datetime(2026, 7, 14, 11, 0, tzinfo=timezone.utc)
CHANNEL = teams_conversation_ref({"id": "19:general@thread.tacv2", "topic": "General", "type": "channel"})


def _parsed(message_id: str = "m1", **fields) -> dict:
    """A message as the Chat Service client parses it."""

    return {
        "id": message_id,
        "conversationid": CHANNEL.conversation_id,
        "from": "Ada",
        "content": "Payroll runs on Fridays.",
        "time": SENT,
        "edited_time": None,
        "rootMessageId": None,
        "parentMessageId": None,
        "mentions": [],
        "attachments": [{"name": "plan.pdf"}],
        "hosted_content_ids": [],
        **fields,
    }


@pytest.mark.parametrize(
    "fields",
    [{}, {"edited_time": EDITED}, {"rootMessageId": "root", "parentMessageId": "root"}],
    ids=["plain", "edited", "reply"],
)
def test_a_record_is_the_record_a_teams_window_kept_for_the_same_message(fields):
    parsed = _parsed(**fields)
    # The message as a stored window package carries it.
    window_message = {
        "id": parsed["id"],
        "from": parsed["from"],
        "content": parsed["content"],
        "time": parsed["time"].isoformat(),
        **({"edited_time": parsed["edited_time"].isoformat()} if parsed.get("edited_time") else {}),
        "mentions": parsed["mentions"],
        "attachments": parsed["attachments"],
        "hosted_content_ids": parsed["hosted_content_ids"],
        "is_root": True,
        **({"parentMessageId": parsed["parentMessageId"]} if parsed.get("parentMessageId") else {}),
        **({"rootMessageId": parsed["rootMessageId"]} if parsed.get("rootMessageId") else {}),
    }
    assert teams_message_record(parsed, CHANNEL) == message_record(
        window_message, conversation_id=CHANNEL.conversation_id, conversation_name="General"
    )


def test_a_reply_names_its_thread_root_and_a_root_names_none():
    reply = teams_provider_message(_parsed("reply", rootMessageId="root"), CHANNEL)
    root = teams_provider_message(_parsed("root", rootMessageId="root"), CHANNEL)

    assert reply.thread_root_id == "root"
    assert reply.record["reply_to_id"] == "root"
    assert root.thread_root_id is None


def test_every_teams_conversation_threads_inline_and_polls_return_replies():
    chat = teams_conversation_ref({"id": "19:chat@thread.v2", "topic": "Payroll", "type": "group_chat", "team_name": None})
    channel = teams_conversation_ref(
        {"id": "19:general@thread.tacv2", "topic": "General", "type": "channel", "team_name": "Payroll"}
    )

    assert {ref.threading for ref in (chat, channel)} == {ThreadingModel.INLINE}
    assert chat.poll_includes_replies and channel.poll_includes_replies
    assert chat.label.title_fields == (("Conversation type", "group_chat"),)
    assert channel.label.title_fields == (("Conversation type", "channel"), ("Team", "Payroll"))
    assert channel.label.message_link.format(message_id="m1") == (
        "https://teams.microsoft.com/l/message/19:general@thread.tacv2/m1"
    )


async def test_a_poll_maps_messages_and_keeps_the_paging_evidence():
    client = TeamsChatClient(region="emea")
    client.read_messages = AsyncMock(
        return_value=TeamsMessagePoll(messages=(_parsed(),), outcome=PollOutcome.REACHED_START, stop_reason="x")
    )
    provider = TeamsConversationProvider(client, poll_budget_seconds=30)

    poll = await provider.poll(CHANNEL, Interval(start=SENT))

    assert [message.message_id for message in poll.messages] == ["m1"]
    assert poll.outcome is PollOutcome.REACHED_START
    client.read_messages.assert_awaited_once_with(
        CHANNEL.conversation_id, start=SENT, end=None, budget_seconds=30
    )


async def test_a_selected_conversation_missing_from_the_recent_listing_is_still_polled():
    status, ref = await TeamsConversationProvider(TeamsChatClient(region="emea")).get_conversation("19:old@thread.v2")

    assert status.access is ConversationAccess.READABLE
    assert ref is not None and ref.label.kind == "group_chat"
