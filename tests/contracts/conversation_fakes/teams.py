"""The Teams Chat Service, faked at the transport of a real ``TeamsChatClient``.

The fake answers the way the Chat Service does:

- message pages run newest first, ``pageSize`` messages to a page;
- ``endTime`` is an inclusive millisecond cursor: a page holds the messages
  placed at or before it;
- each page certifies the time it is complete from in
  ``_metadata.lastCompleteSegmentStartTime``, the millisecond of its oldest
  message, so the next page repeats that millisecond;
- the conversation's oldest page is certified from the epoch;
- a deleted message keeps its creation time but loses its content, and is
  placed where it was deleted, so a newer page can hold a far older message;
- 401, 403, 404, 5xx and malformed pages are answered as named answers.

The page helpers (:func:`chat_message`, :func:`message_page`) and
:class:`FakeChatService` also serve the Teams client's own tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx

from memforge.auth.teams_browser_session import CHAT_API_AUDIENCE, GRAPH_API_AUDIENCE
from memforge.genes.teams_client import TeamsChatClient
from memforge.genes.teams_provider import DEFAULT_POLL_BUDGET_SECONDS, TeamsConversationProvider
from tests.contracts.conversation_fakes import (
    NOW,
    ConversationProviderFakes,
    FakeConversationConnection,
    Scenario,
    ScenarioMessage,
)

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
CHAT = "19:contract@thread.v2"
ONE_MILLISECOND = timedelta(milliseconds=1)

# The tokens the keychain holds for a connected Teams session.
TOKENS = {
    CHAT_API_AUDIENCE: {"token": "chat-token", "expiresAt": 4_102_444_800},
    GRAPH_API_AUDIENCE: {"token": "graph-token", "expiresAt": 4_102_444_800},
}

Answer = Callable[[httpx.Request], httpx.Response]


def milliseconds(at: datetime) -> int:
    """``at`` as the Chat Service states times: whole milliseconds since the epoch."""

    return (at - EPOCH) // ONE_MILLISECOND


def chat_message(
    message_id: str,
    created_at: datetime,
    *,
    conversation_id: str = CHAT,
    reply_to: str | None = None,
    deleted_at: datetime | None = None,
) -> dict:
    """One raw Chat Service message; a deleted one keeps its id and time and loses its content."""

    message: dict = {
        "id": message_id,
        "conversationid": conversation_id,
        "imdisplayname": "Ada",
        "content": "" if deleted_at else f"<p>{message_id}</p>",
        "messagetype": "RichText/Html",
        "composetime": _composetime(created_at),
    }
    if reply_to:
        message["rootMessageId"] = reply_to
    if deleted_at:
        message["properties"] = {"deletetime": milliseconds(deleted_at)}
    return message


def message_page(*messages: dict, complete_from: datetime | None = EPOCH) -> dict:
    """A message page certifying the time it is complete from, by default the oldest page's epoch.

    ``complete_from=None`` leaves the certificate out.
    """

    body: dict = {"messages": list(messages)}
    if complete_from is not None:
        body["_metadata"] = {"lastCompleteSegmentStartTime": milliseconds(complete_from)}
    return body


class FakeChatService:
    """The Chat Service and Microsoft Graph behind one client's transport.

    ``answer`` answers every request; the service records each request, and
    each wait the client asks its stub sleep for. ``now`` is the client's
    clock, which stands still unless a test moves it. A client reads
    ``tokens`` as its keychain, by default a session with Graph access.
    """

    def __init__(self, answer: Answer) -> None:
        self._answer = answer
        self.requests: list[httpx.Request] = []
        self.waits: list[float] = []
        self.now = 0.0

    def client(self, region: str = "emea", *, tokens: dict = TOKENS) -> TeamsChatClient:
        return TeamsChatClient(
            region,
            transport=httpx.MockTransport(self._handle),
            sleep=self._sleep,
            clock=lambda: self.now,
            load_tokens=lambda: tokens,
        )

    async def _sleep(self, seconds: float) -> None:
        self.waits.append(seconds)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._answer(request)


def serving(*answers: dict | httpx.Response) -> FakeChatService:
    """A service answering each request with the next of ``answers``; a body is answered with HTTP 200."""

    queue = list(answers)

    def answer(request: httpx.Request) -> httpx.Response:
        if not queue:
            raise AssertionError(f"the fake Chat Service has no answer left for {request.url}")
        next_answer = queue.pop(0)
        return next_answer if isinstance(next_answer, httpx.Response) else httpx.Response(200, json=next_answer)

    return FakeChatService(answer)


ANSWERS: dict[str, Answer] = {
    "HTTP 401": lambda _request: httpx.Response(401, json={"errorCode": 911, "message": "Authentication failed."}),
    "HTTP 403": lambda _request: httpx.Response(403, json={"errorCode": 403, "message": "Forbidden"}),
    "HTTP 404": lambda _request: httpx.Response(404, json={"errorCode": 404, "message": "Not found"}),
    "HTTP 500": lambda _request: httpx.Response(500),
    "HTTP 503": lambda _request: httpx.Response(503),
    "a page without a messages list": lambda _request: httpx.Response(
        200, json={"error": {"code": "TransientShapeDrift"}}
    ),
    "a page that certifies no complete time": lambda _request: httpx.Response(
        200, json=message_page(chat_message("uncertified", NOW - timedelta(hours=1)), complete_from=None)
    ),
}


class ScenarioChatService(FakeChatService):
    """A Chat Service holding one conversation's scenario, or giving one named answer to every request."""

    def __init__(self, scenario: Scenario, *, answer: str | None = None, budget_seconds: float) -> None:
        super().__init__(ANSWERS[answer] if answer else self._serve)
        self._scenario = scenario
        self._budget_seconds = budget_seconds
        self._clock_step = 0.0

    def stop_after(self, pages: int) -> None:
        """Move the clock with every page served, so a poll's budget runs out once it has read ``pages``."""

        self._clock_step = self._budget_seconds / pages * (1 + 1e-9)

    def _serve(self, request: httpx.Request) -> httpx.Response:
        segments = request.url.path.split("/")
        if segments[-1] == "conversations":
            return httpx.Response(200, json={"conversations": [_listed(CHAT)]})
        if segments[-1] == "messages" and segments[-2] == CHAT:
            return httpx.Response(200, json=self._page(request.url.params.get("endTime")))
        return httpx.Response(404, json={"errorCode": 404, "message": "Not found"})

    def _page(self, end_time: str | None) -> dict:
        held = sorted(
            (message for message in self._scenario.messages if end_time is None or _placed(message) <= int(end_time)),
            key=lambda message: (_placed(message), message.message_id),
            reverse=True,
        )
        page, older = held[: self._scenario.page_size], held[self._scenario.page_size :]
        if not page:
            return message_page()
        self.now += self._clock_step
        messages = [_raw(message) for message in page]
        if older:
            certified = EPOCH + _placed(page[-1]) * ONE_MILLISECOND
        elif self._scenario.ends_on_empty_page:
            certified = EPOCH + (_placed(page[-1]) - 1) * ONE_MILLISECOND
        else:
            certified = EPOCH
        return message_page(*messages, complete_from=certified)


class TeamsFakes(ConversationProviderFakes):
    conversation_id = CHAT
    time_precision = ONE_MILLISECOND
    refused_answers = ("HTTP 403", "HTTP 404")
    failing_answers = (
        "HTTP 401",
        "HTTP 500",
        "HTTP 503",
        "a page without a messages list",
        "a page that certifies no complete time",
    )

    @asynccontextmanager
    async def connect(
        self, scenario: Scenario, *, answer: str | None = None
    ) -> AsyncIterator[FakeConversationConnection]:
        service = ScenarioChatService(scenario, answer=answer, budget_seconds=DEFAULT_POLL_BUDGET_SECONDS)
        client = service.client()
        try:
            yield FakeConversationConnection(
                provider=TeamsConversationProvider(client, poll_budget_seconds=DEFAULT_POLL_BUDGET_SECONDS),
                stop_after=service.stop_after,
            )
        finally:
            await client.close()


FAKES = TeamsFakes()


def _composetime(at: datetime) -> str:
    utc = at.astimezone(timezone.utc)
    return f"{utc:%Y-%m-%dT%H:%M:%S}.{utc.microsecond // 1000:03d}Z"


def _placed(message: ScenarioMessage) -> int:
    """The millisecond the Chat Service places a message at: where it was deleted, else when it was created."""

    return milliseconds(message.deleted_at or message.created_at)


def _raw(message: ScenarioMessage) -> dict:
    return chat_message(
        message.message_id, message.created_at, reply_to=message.reply_to, deleted_at=message.deleted_at
    )


def _listed(conversation_id: str) -> dict:
    return {
        "id": conversation_id,
        "threadProperties": {"topic": "Contract chat"},
        "lastMessage": {"composetime": _composetime(NOW)},
    }

