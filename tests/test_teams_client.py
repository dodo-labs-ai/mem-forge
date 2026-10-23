"""The Teams Chat Service client: requests, parsing, and the evidence each poll's paging gives."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from memforge.conversations.port import PollOutcome
from memforge.genes.teams_client import (
    AuthenticationError,
    TeamsChatClient,
    TeamsMessagePageSchemaError,
    TeamsMessagePaginationError,
)
from memforge.source_artifacts import SourceArtifactContractError
from memforge.auth.teams_browser_session import CHAT_API_AUDIENCE
from tests.contracts.conversation_fakes.teams import (
    CHAT,
    TOKENS,
    FakeChatService,
    chat_message,
    message_page,
    milliseconds,
    serving,
)

NOW = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)


def _page_params(service, index: int) -> dict[str, str]:
    """The query of the ``index``-th request the client sent the fake service."""

    return dict(service.requests[index].url.params)


def test_parse_conversations_preserves_channel_team_identity():
    client = TeamsChatClient(region="emea")

    [conversation] = client._parse_conversations(
        {
            "conversations": [
                {
                    "id": "19:channel@thread.tacv2",
                    "threadProperties": {
                        "topic": "Architecture",
                        "groupId": "team-1",
                    },
                    "lastActivity": "2026-04-15T12:00:00Z",
                }
            ]
        }
    )

    assert conversation["type"] == "channel"
    assert conversation["team_id"] == "team-1"


async def test_request_raises_auth_error_without_retry_on_401():
    service = serving(httpx.Response(401, json={"errorCode": 911, "message": "Authentication failed."}))

    with pytest.raises(AuthenticationError, match="Teams session expired"):
        await service.client().validate()

    assert len(service.requests) == 1


async def test_request_retries_remote_protocol_error():
    def answer(request: httpx.Request) -> httpx.Response:
        if len(service.requests) == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=request)
        return httpx.Response(200, json={"conversations": []})

    service = FakeChatService(answer)

    await service.client().validate()

    assert len(service.requests) == 2
    assert len(service.waits) == 1


async def test_validate_retries_a_failed_probe():
    service = serving(httpx.Response(503), {"conversations": []})

    await service.client().validate()

    assert [request.url.path for request in service.requests] == ["/api/chatsvc/emea/v1/users/ME/conversations"] * 2
    assert _page_params(service, 1) == {"pageSize": "1"}
    assert service.waits == [2.0]


def test_parse_message_preserves_real_rest_root_message_id():
    client = TeamsChatClient(region="emea")

    parsed = client._parse_message(
        {
            "id": "reply-1",
            "rootMessageId": "root-1",
            "conversationid": "19:channel@example",
            "imdisplayname": "Alice",
            "content": "<p>Confirmed</p>",
            "messagetype": "RichText/Html",
            "composetime": "2026-04-15T12:00:00Z",
        }
    )

    assert parsed is not None
    assert parsed["id"] == "reply-1"
    assert parsed["rootMessageId"] == "root-1"
    assert parsed["conversationid"] == "19:channel@example"


def test_parse_message_preserves_only_official_hosted_content_ids():
    client = TeamsChatClient(region="emea")

    parsed = client._parse_message(
        {
            "id": "message-1",
            "conversationid": "19:chat@example",
            "imdisplayname": "Alice",
            "content": (
                "<p>Architecture:</p>"
                '<img src="https://graph.microsoft.com/v1.0/chats/19:chat@example/'
                'messages/message-1/hostedContents/hosted-1/$value">'
                '<img src="../hostedContents/hosted-2/$value">'
                '<img src="https://example.test/unrelated.png">'
            ),
            "messagetype": "RichText/Html",
            "composetime": "2026-04-15T12:00:00Z",
        }
    )

    assert parsed is not None
    assert parsed["hosted_content_ids"] == ["hosted-1", "hosted-2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("conversation_type", "team_id", "root_message_id", "expected_path"),
    [
        (
            "group_chat",
            None,
            None,
            "/chats/19%3Achat%40example/messages/message-1/hostedContents/hosted-1/$value",
        ),
        (
            "channel",
            "team-1",
            "message-1",
            "/teams/team-1/channels/19%3Achannel%40example/messages/message-1/hostedContents/hosted-1/$value",
        ),
        (
            "channel",
            "team-1",
            "root-1",
            "/teams/team-1/channels/19%3Achannel%40example/messages/root-1/replies/message-1/"
            "hostedContents/hosted-1/$value",
        ),
    ],
)
async def test_get_hosted_content_uses_exact_graph_value_route(
    conversation_type,
    team_id,
    root_message_id,
    expected_path,
):
    service = serving(httpx.Response(200, content=b"\x89PNG\r\n\x1a\nhosted", headers={"content-type": "image/png"}))

    content, media_type = await service.client().get_hosted_content(
        conversation_id=("19:channel@example" if conversation_type == "channel" else "19:chat@example"),
        message_id="message-1",
        hosted_content_id="hosted-1",
        conversation_type=conversation_type,
        team_id=team_id,
        root_message_id=root_message_id,
    )

    assert content == b"\x89PNG\r\n\x1a\nhosted"
    assert media_type == "image/png"
    [request] = service.requests
    assert request.url.host == "graph.microsoft.com"
    assert request.url.raw_path.decode() == f"/v1.0{expected_path}"
    assert request.headers["Authorization"] == "Bearer graph-token"


async def test_reading_an_inline_image_requires_graph_access_before_any_request():
    service = serving()
    client = service.client(tokens={CHAT_API_AUDIENCE: TOKENS[CHAT_API_AUDIENCE]})

    with pytest.raises(AuthenticationError, match="Teams hosted-content requires Microsoft Graph access"):
        await client.get_hosted_content(
            conversation_id="19:chat@thread.v2",
            message_id="m1",
            hosted_content_id="h1",
            conversation_type="group_chat",
            team_id=None,
        )

    assert service.requests == []
    await client.close()


@pytest.mark.asyncio
async def test_request_bytes_stops_before_buffering_oversized_body():
    client = TeamsChatClient(region="emea")
    chunks_read = 0

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "image/png"}
        request = httpx.Request("GET", "https://graph.microsoft.com/hosted")

        async def aiter_bytes(self):
            nonlocal chunks_read
            for chunk in (b"1234", b"5678", b"not-read"):
                chunks_read += 1
                yield chunk

        def raise_for_status(self):
            return None

    class FakeStream:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    class FakeClient:
        def stream(self, method: str, url: str):
            assert method == "GET"
            assert url == "/hosted"
            return FakeStream()

    with pytest.raises(SourceArtifactContractError, match="storage limit"):
        await client._request_bytes(FakeClient(), "/hosted", max_bytes=6)

    assert chunks_read == 2


async def test_a_page_reaching_back_past_the_start_keeps_newer_messages_and_proves_the_start():
    client = serving(message_page(chat_message("new", NOW), chat_message("old", NOW - timedelta(days=2)))).client()

    poll = await client.read_messages(CHAT, start=NOW - timedelta(days=1), end=None, budget_seconds=60)

    assert [message["id"] for message in poll.messages] == ["new"]
    assert poll.outcome is PollOutcome.REACHED_START
    assert poll.earliest_complete_page_time == NOW - timedelta(days=2)


async def test_running_out_of_pages_before_the_start_is_history_exhausted():
    client = serving(message_page(chat_message("newest", NOW))).client()

    poll = await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)

    assert [message["id"] for message in poll.messages] == ["newest"]
    assert poll.outcome is PollOutcome.HISTORY_EXHAUSTED
    assert poll.earliest_complete_page_time == NOW


@pytest.mark.parametrize("last_page", ["epoch", "empty"])
async def test_running_out_of_history_names_the_time_its_read_is_complete_from(last_page):
    certified = NOW - timedelta(hours=1)
    first = NOW - timedelta(hours=3)
    newer = message_page(
        chat_message("newest", NOW),
        chat_message("deleted-later", NOW - timedelta(days=3), deleted_at=NOW - timedelta(minutes=30)),
        complete_from=certified,
    )
    epoch = message_page(chat_message("older", NOW - timedelta(hours=2)), chat_message("first", first))
    service = serving(newer, epoch if last_page == "epoch" else message_page())

    poll = await service.client().read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)

    assert poll.outcome is PollOutcome.HISTORY_EXHAUSTED
    assert poll.earliest_complete_page_time == (first if last_page == "epoch" else certified)
    assert len(service.requests) == 2


async def test_pages_follow_the_certified_time_cursor_and_repeat_the_boundary_millisecond(monkeypatch):
    monkeypatch.setattr("memforge.genes.teams_client._MAX_PAGE_SIZE", 2)
    older = NOW - timedelta(hours=1)
    before = NOW - timedelta(days=15)
    service = serving(
        message_page(chat_message("newest", NOW), chat_message("older", older), complete_from=older),
        message_page(
            chat_message("older", older),
            chat_message("same-millisecond", older),
            chat_message("before", before),
            complete_from=before,
        ),
    )

    poll = await service.client().read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)

    assert [message["id"] for message in poll.messages] == ["newest", "older", "same-millisecond"]
    assert poll.outcome is PollOutcome.REACHED_START
    assert _page_params(service, 1) == {
        "pageSize": "2",
        "view": "msnp24Equivalent",
        "startTime": "0",
        "endTime": str(milliseconds(older)),
    }
    assert service.waits == [1.0]


async def test_a_backfill_starts_just_before_its_end():
    end = NOW - timedelta(days=3)
    service = serving(
        message_page(chat_message("inside", end - timedelta(hours=1)), chat_message("past", end - timedelta(days=5)))
    )

    poll = await service.client().read_messages(CHAT, start=end - timedelta(days=2), end=end, budget_seconds=60)

    assert [message["id"] for message in poll.messages] == ["inside"]
    assert _page_params(service, 0)["endTime"] == str(milliseconds(end) - 1)


async def test_a_backfill_ending_just_after_a_page_time_reads_that_millisecond():
    page_time = datetime(2026, 10, 20, 9, 0, 0, 789000, tzinfo=timezone.utc)
    service = serving(
        message_page(
            chat_message("at-page-time", page_time), chat_message("past", page_time - timedelta(days=5))
        )
    )

    poll = await service.client().read_messages(
        CHAT, start=page_time - timedelta(days=2), end=page_time + timedelta(microseconds=1), budget_seconds=60
    )

    assert [message["id"] for message in poll.messages] == ["at-page-time"]
    assert _page_params(service, 0)["endTime"] == str(milliseconds(page_time))


async def test_a_spent_time_budget_stops_early_after_the_last_complete_page(monkeypatch):
    monkeypatch.setattr("memforge.genes.teams_client._MAX_PAGE_SIZE", 1)
    certified = NOW - timedelta(hours=1)
    service = serving(
        message_page(
            chat_message("newest", NOW),
            chat_message("older-than-its-page", NOW - timedelta(days=5)),
            complete_from=certified,
        ),
        message_page(chat_message("older", NOW - timedelta(hours=2))),
    )

    poll = await service.client().read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=0)

    assert {message["id"] for message in poll.messages} == {"newest", "older-than-its-page"}
    assert poll.outcome is PollOutcome.STOPPED_EARLY
    assert poll.earliest_complete_page_time == certified
    assert len(service.requests) == 1


async def test_a_deleted_older_message_on_a_newer_page_does_not_move_the_cursor(monkeypatch):
    monkeypatch.setattr("memforge.genes.teams_client._MAX_PAGE_SIZE", 3)
    hour_ago = NOW - timedelta(hours=1)
    deleted = chat_message("deleted", NOW - timedelta(days=20), deleted_at=NOW - timedelta(minutes=5))
    service = serving(
        message_page(chat_message("newest", NOW), deleted, chat_message("hour-ago", hour_ago), complete_from=hour_ago),
        message_page(chat_message("hour-ago", hour_ago), chat_message("yesterday", NOW - timedelta(days=1))),
    )

    poll = await service.client().read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)

    assert [message["id"] for message in poll.messages] == ["newest", "hour-ago", "yesterday"]
    assert poll.outcome is PollOutcome.HISTORY_EXHAUSTED
    assert _page_params(service, 1)["endTime"] == str(milliseconds(hour_ago))


async def test_a_page_that_certifies_no_complete_start_fails_the_poll():
    client = serving(message_page(chat_message("newest", NOW), complete_from=None)).client()

    with pytest.raises(TeamsMessagePageSchemaError, match="does not certify"):
        await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)


async def test_a_page_certified_from_the_epoch_is_the_oldest_and_bounded_by_its_oldest_message():
    service = serving(message_page(chat_message("newest", NOW), chat_message("first", NOW - timedelta(days=2))))

    poll = await service.client().read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)

    assert [message["id"] for message in poll.messages] == ["newest", "first"]
    assert poll.outcome is PollOutcome.HISTORY_EXHAUSTED
    assert len(service.requests) == 1


async def test_a_short_page_certified_from_a_later_time_is_not_the_oldest():
    certified = NOW - timedelta(hours=1)
    service = serving(
        message_page(
            chat_message("newest", NOW),
            chat_message("deleted-later", NOW - timedelta(days=20), deleted_at=NOW - timedelta(minutes=30)),
            complete_from=certified,
        ),
        message_page(),
    )

    poll = await service.client().read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)

    assert poll.outcome is PollOutcome.HISTORY_EXHAUSTED
    assert _page_params(service, 1)["endTime"] == str(milliseconds(certified))


async def test_a_time_cursor_that_does_not_advance_fails_the_poll(monkeypatch):
    monkeypatch.setattr("memforge.genes.teams_client._MAX_PAGE_SIZE", 1)
    client = serving(
        message_page(chat_message("same", NOW), complete_from=NOW),
        message_page(chat_message("same", NOW), complete_from=NOW),
    ).client()

    with pytest.raises(TeamsMessagePaginationError, match="did not advance"):
        await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)


async def test_a_page_without_a_messages_list_fails_the_poll():
    client = serving({"error": {"code": "TransientShapeDrift"}}).client()

    with pytest.raises(ValueError, match="explicit messages list"):
        await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)


async def test_a_content_message_without_a_source_time_fails_the_poll():
    client = serving(message_page({"id": "m1", "content": "still present"})).client()

    with pytest.raises(ValueError, match="stable id or source timestamp"):
        await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)
