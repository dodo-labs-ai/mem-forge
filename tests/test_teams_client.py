"""The Teams Chat Service client: requests, parsing, and the evidence each poll's paging gives."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from memforge.conversations.port import PollOutcome
from memforge.genes.teams_client import (
    AuthenticationError,
    TeamsChatClient,
    TeamsMessagePageSchemaError,
    TeamsMessagePaginationError,
)
from memforge.source_artifacts import MAX_SOURCE_ARTIFACT_STORAGE_BYTES, SourceArtifactContractError

NOW = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
CHAT = "19:conversation@thread.v2"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _message(message_id: str, created_at: datetime) -> dict:
    return {
        "id": message_id,
        "conversationid": CHAT,
        "content": f"<p>{message_id}</p>",
        "messagetype": "RichText/Html",
        "composetime": created_at.isoformat(),
    }


def _page(*messages: dict, complete_from: datetime | None = EPOCH) -> MagicMock:
    """A Chatsvc page certifying the segment start it is complete from, by default the oldest page's epoch.

    ``complete_from=None`` leaves the certificate out.
    """

    response = MagicMock()
    body: dict = {"messages": list(messages)}
    if complete_from is not None:
        body["_metadata"] = {"lastCompleteSegmentStartTime": _milliseconds(complete_from)}
    response.json.return_value = body
    return response


def _milliseconds(at: datetime) -> int:
    return int(at.timestamp() * 1000)


def _client(*pages: MagicMock) -> TeamsChatClient:
    client = TeamsChatClient(region="emea")
    client._ensure_clients = AsyncMock()
    client._chat_client = object()
    client._request = AsyncMock(side_effect=list(pages))
    return client


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


@pytest.mark.asyncio
async def test_request_raises_auth_error_without_retry_on_401():
    client = TeamsChatClient(region="emea")
    calls = 0

    class FakeClient:
        async def request(self, method: str, url: str, **kwargs):
            nonlocal calls
            calls += 1
            request = httpx.Request(method, f"https://teams.cloud.microsoft{url}")
            return httpx.Response(
                401,
                json={"errorCode": 911, "message": "Authentication failed."},
                request=request,
            )

    with pytest.raises(AuthenticationError, match="Teams session expired"):
        await client._request(FakeClient(), "GET", "/conversations")

    assert calls == 1


@pytest.mark.asyncio
async def test_request_retries_remote_protocol_error(monkeypatch):
    client = TeamsChatClient(region="emea")
    calls = 0

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("memforge.pipeline.retry.asyncio.sleep", no_sleep)

    class FakeClient:
        async def request(self, method: str, url: str, **kwargs):
            nonlocal calls
            calls += 1
            request = httpx.Request(method, f"https://teams.cloud.microsoft{url}")
            if calls == 1:
                raise httpx.RemoteProtocolError(
                    "Server disconnected without sending a response.",
                    request=request,
                )
            return httpx.Response(200, json={"conversations": []}, request=request)

    response = await client._request(FakeClient(), "GET", "/conversations")

    assert response.status_code == 200
    assert calls == 2


@pytest.mark.asyncio
async def test_validate_uses_retrying_request():
    client = TeamsChatClient(region="emea")
    client._ensure_clients = AsyncMock()
    client._chat_client = object()
    client._request = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"conversations": []},
            request=httpx.Request("GET", "https://teams.cloud.microsoft/conversations"),
        )
    )

    await client.validate()

    client._request.assert_awaited_once()


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
    client = TeamsChatClient(region="emea")
    client._ensure_clients = AsyncMock()
    client._graph_client = object()
    client._request_bytes = AsyncMock(
        return_value=(
            b"\x89PNG\r\n\x1a\nhosted",
            {"content-type": "image/png", "content-length": "14"},
        )
    )

    content, media_type = await client.get_hosted_content(
        conversation_id=("19:channel@example" if conversation_type == "channel" else "19:chat@example"),
        message_id="message-1",
        hosted_content_id="hosted-1",
        conversation_type=conversation_type,
        team_id=team_id,
        root_message_id=root_message_id,
    )

    assert content == b"\x89PNG\r\n\x1a\nhosted"
    assert media_type == "image/png"
    client._request_bytes.assert_awaited_once_with(
        client._graph_client,
        expected_path,
        max_bytes=MAX_SOURCE_ARTIFACT_STORAGE_BYTES,
    )


@pytest.mark.asyncio
async def test_get_hosted_content_requires_graph_capability_before_network_request(
    monkeypatch,
):
    from memforge.auth.teams_auth import CHAT_API_AUDIENCE, TeamsAuthenticator

    monkeypatch.setattr(
        TeamsAuthenticator,
        "load_tokens",
        staticmethod(
            lambda: {
                CHAT_API_AUDIENCE: {
                    "token": "chat-token",
                    "expiresAt": 4_102_444_800,
                }
            }
        ),
    )
    client = TeamsChatClient(region="emea")
    client._request_bytes = AsyncMock(
        side_effect=AssertionError(
            "hosted-content retrieval must not send a Chat token to Microsoft Graph"
        )
    )

    with pytest.raises(
        AuthenticationError,
        match="requires Microsoft Graph access",
    ):
        await client.get_hosted_content(
            conversation_id="19:chat@example",
            message_id="message-1",
            hosted_content_id="hosted-1",
            conversation_type="group_chat",
            team_id=None,
        )

    client._request_bytes.assert_not_awaited()
    await client.close()


@pytest.mark.asyncio
async def test_reading_an_inline_image_requires_graph_access(
    monkeypatch,
):
    from memforge.auth.teams_auth import CHAT_API_AUDIENCE, TeamsAuthenticator

    monkeypatch.setattr(
        TeamsAuthenticator,
        "load_tokens",
        staticmethod(
            lambda: {
                CHAT_API_AUDIENCE: {
                    "token": "chat-token",
                    "expiresAt": 4_102_444_800,
                }
            }
        ),
    )
    client = TeamsChatClient(region="emea")

    with pytest.raises(
        AuthenticationError,
        match="Teams hosted-content requires Microsoft Graph access",
    ):
        await client.get_hosted_content(
            conversation_id="19:chat@thread.v2",
            message_id="m1",
            hosted_content_id="h1",
            conversation_type="group_chat",
            team_id=None,
        )

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
    client = _client(_page(_message("old", NOW - timedelta(days=2)), _message("new", NOW)))

    poll = await client.read_messages(CHAT, start=NOW - timedelta(days=1), end=None, budget_seconds=60)

    assert [message["id"] for message in poll.messages] == ["new"]
    assert poll.outcome is PollOutcome.REACHED_START
    assert poll.earliest_complete_page_time == NOW - timedelta(days=2)


async def test_running_out_of_pages_before_the_start_is_history_exhausted(monkeypatch):
    monkeypatch.setattr("memforge.genes.teams_client.asyncio.sleep", AsyncMock())
    client = _client(_page(_message("newest", NOW)))

    poll = await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)

    assert [message["id"] for message in poll.messages] == ["newest"]
    assert poll.outcome is PollOutcome.HISTORY_EXHAUSTED
    assert poll.earliest_complete_page_time == NOW


@pytest.mark.parametrize("last_page", ["epoch", "empty"])
async def test_running_out_of_history_names_the_time_its_read_is_complete_from(monkeypatch, last_page):
    monkeypatch.setattr("memforge.genes.teams_client.asyncio.sleep", AsyncMock())
    certified = NOW - timedelta(hours=1)
    first = NOW - timedelta(hours=3)
    newer = _page(_message("newest", NOW), _message("deleted-later", NOW - timedelta(days=3)), complete_from=certified)
    epoch = _page(_message("older", NOW - timedelta(hours=2)), _message("first", first))
    oldest = epoch if last_page == "epoch" else _page()
    client = _client(newer, oldest)

    poll = await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)

    assert poll.outcome is PollOutcome.HISTORY_EXHAUSTED
    assert poll.earliest_complete_page_time == (first if last_page == "epoch" else certified)
    assert client._request.await_count == 2


async def test_pages_follow_the_certified_time_cursor_and_repeat_the_boundary_millisecond(monkeypatch):
    monkeypatch.setattr("memforge.genes.teams_client._MAX_PAGE_SIZE", 2)
    monkeypatch.setattr("memforge.genes.teams_client.asyncio.sleep", AsyncMock())
    older = NOW - timedelta(hours=1)
    before = NOW - timedelta(days=15)
    client = _client(
        _page(_message("newest", NOW), _message("older", older), complete_from=older),
        _page(
            _message("older", older),
            _message("same-millisecond", older),
            _message("before", before),
            complete_from=before,
        ),
    )

    poll = await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)

    assert [message["id"] for message in poll.messages] == ["newest", "older", "same-millisecond"]
    assert poll.outcome is PollOutcome.REACHED_START
    assert client._request.await_args_list[1].kwargs["params"] == {
        "pageSize": 2,
        "view": "msnp24Equivalent",
        "startTime": 0,
        "endTime": _milliseconds(older),
    }


async def test_a_backfill_starts_just_before_its_end(monkeypatch):
    end = NOW - timedelta(days=3)
    client = _client(_page(_message("inside", end - timedelta(hours=1)), _message("past", end - timedelta(days=5))))

    poll = await client.read_messages(CHAT, start=end - timedelta(days=2), end=end, budget_seconds=60)

    assert [message["id"] for message in poll.messages] == ["inside"]
    assert client._request.await_args.kwargs["params"]["endTime"] == int(end.timestamp() * 1000) - 1


async def test_a_backfill_ending_just_after_a_page_time_reads_that_millisecond():
    page_time = datetime(2026, 10, 20, 9, 0, 0, 789000, tzinfo=timezone.utc)
    client = _client(_page(_message("at-page-time", page_time), _message("past", page_time - timedelta(days=5))))

    poll = await client.read_messages(
        CHAT, start=page_time - timedelta(days=2), end=page_time + timedelta(microseconds=1), budget_seconds=60
    )

    assert [message["id"] for message in poll.messages] == ["at-page-time"]
    assert client._request.await_args.kwargs["params"]["endTime"] == int(page_time.timestamp() * 1000)


async def test_a_spent_time_budget_stops_early_after_the_last_complete_page(monkeypatch):
    monkeypatch.setattr("memforge.genes.teams_client._MAX_PAGE_SIZE", 1)
    monkeypatch.setattr("memforge.genes.teams_client.asyncio.sleep", AsyncMock())
    certified = NOW - timedelta(hours=1)
    client = _client(
        _page(_message("newest", NOW), _message("deleted-later", NOW - timedelta(days=5)), complete_from=certified),
        _page(_message("older", NOW - timedelta(hours=2))),
    )

    poll = await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=0)

    assert {message["id"] for message in poll.messages} == {"newest", "deleted-later"}
    assert poll.outcome is PollOutcome.STOPPED_EARLY
    assert poll.earliest_complete_page_time == certified
    assert client._request.await_count == 1


async def test_a_deleted_older_message_on_a_newer_page_does_not_move_the_cursor(monkeypatch):
    monkeypatch.setattr("memforge.genes.teams_client._MAX_PAGE_SIZE", 3)
    monkeypatch.setattr("memforge.genes.teams_client.asyncio.sleep", AsyncMock())
    hour_ago = NOW - timedelta(hours=1)
    deleted = {
        **_message("deleted", NOW - timedelta(days=20)),
        "content": "",
        "properties": {"deletetime": _milliseconds(NOW - timedelta(minutes=5))},
    }
    client = _client(
        _page(_message("newest", NOW), deleted, _message("hour-ago", hour_ago), complete_from=hour_ago),
        _page(_message("hour-ago", hour_ago), _message("yesterday", NOW - timedelta(days=1))),
    )

    poll = await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)

    assert {message["id"] for message in poll.messages} >= {"newest", "hour-ago", "yesterday"}
    assert poll.outcome is PollOutcome.HISTORY_EXHAUSTED
    assert client._request.await_args_list[1].kwargs["params"]["endTime"] == _milliseconds(hour_ago)


async def test_a_page_that_certifies_no_complete_start_fails_the_poll():
    client = _client(_page(_message("newest", NOW), complete_from=None))

    with pytest.raises(TeamsMessagePageSchemaError, match="does not certify"):
        await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)


async def test_a_page_certified_from_the_epoch_is_the_oldest_and_bounded_by_its_oldest_message():
    client = _client(_page(_message("newest", NOW), _message("first", NOW - timedelta(days=2))))

    poll = await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)

    assert [message["id"] for message in poll.messages] == ["newest", "first"]
    assert poll.outcome is PollOutcome.HISTORY_EXHAUSTED
    assert client._request.await_count == 1


async def test_a_short_page_certified_from_a_later_time_is_not_the_oldest(monkeypatch):
    monkeypatch.setattr("memforge.genes.teams_client.asyncio.sleep", AsyncMock())
    certified = NOW - timedelta(hours=1)
    client = _client(
        _page(_message("newest", NOW), _message("deleted-later", NOW - timedelta(days=20)), complete_from=certified),
        _page(),
    )

    poll = await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)

    assert poll.outcome is PollOutcome.HISTORY_EXHAUSTED
    assert client._request.await_args_list[1].kwargs["params"]["endTime"] == _milliseconds(certified)


async def test_a_time_cursor_that_does_not_advance_fails_the_poll(monkeypatch):
    monkeypatch.setattr("memforge.genes.teams_client._MAX_PAGE_SIZE", 1)
    monkeypatch.setattr("memforge.genes.teams_client.asyncio.sleep", AsyncMock())
    client = _client(_page(_message("same", NOW), complete_from=NOW), _page(_message("same", NOW), complete_from=NOW))

    with pytest.raises(TeamsMessagePaginationError, match="did not advance"):
        await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)


async def test_a_page_without_a_messages_list_fails_the_poll():
    response = MagicMock()
    response.json.return_value = {"error": {"code": "TransientShapeDrift"}}
    client = _client(response)

    with pytest.raises(ValueError, match="explicit messages list"):
        await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)


async def test_a_content_message_without_a_source_time_fails_the_poll():
    client = _client(_page({"id": "m1", "content": "still present"}))

    with pytest.raises(ValueError, match="stable id or source timestamp"):
        await client.read_messages(CHAT, start=NOW - timedelta(days=14), end=None, budget_seconds=60)
