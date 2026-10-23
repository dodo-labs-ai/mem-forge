"""ConnectorHttpClient: when it retries, how long it waits, when it gives up, and how its hooks see each attempt."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from memforge.genes.atlassian_auth import AtlassianRequestLimiter
from memforge.genes.toolkit.http import ConnectorHttpClient, ConnectorUnreachableError, RetryPolicy
from tests.connector_http_fixture import FakeConnectorHttp

BASE_URL = "https://provider.example.test"


def _answers(*responses: httpx.Response | Exception):
    """A handler that answers each request with the next response, or raises the next error; it records the requests."""
    queue = list(responses)
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    return handle, requests


def _http(*responses: httpx.Response | Exception) -> tuple[FakeConnectorHttp, list[httpx.Request]]:
    handle, requests = _answers(*responses)
    return FakeConnectorHttp(handle), requests


def _http_date(seconds_from_now: float) -> str:
    return format_datetime(datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now), usegmt=True)


class RecordingLimiter:
    """A limiter that lets every attempt through and records the spacing each response asked for."""

    def __init__(self) -> None:
        self.attempts = 0
        self.delays: list[float | None] = []

    async def run(self, operation, *, delay_for_response=None, sleep=None):
        self.attempts += 1
        response = await operation()
        self.delays.append(delay_for_response(response) if delay_for_response else None)
        return response


class FakeClock:
    """A monotonic clock that only the client's sleep advances."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


# ---------------------------------------------------------------------------
# Waiting for 429 and 503
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
async def test_a_status_that_asks_to_wait_is_retried_after_retry_after_seconds(status) -> None:
    http, requests = _http(httpx.Response(status, headers={"Retry-After": "5"}), httpx.Response(200, json={"ok": True}))

    async with http.client(BASE_URL) as client:
        response = await client.get("/items")

    assert response.json() == {"ok": True}
    assert len(requests) == 2
    assert http.waits == [5.0]


@pytest.mark.asyncio
async def test_retry_after_as_an_http_date_waits_until_that_time() -> None:
    http, _requests = _http(httpx.Response(503, headers={"Retry-After": _http_date(30)}), httpx.Response(200))

    async with http.client(BASE_URL) as client:
        await client.get("/items")

    [wait] = http.waits
    assert 25.0 <= wait <= 30.0


@pytest.mark.asyncio
async def test_retry_after_in_the_past_retries_at_once() -> None:
    http, _requests = _http(httpx.Response(429, headers={"Retry-After": _http_date(-30)}), httpx.Response(200))

    async with http.client(BASE_URL) as client:
        await client.get("/items")

    assert http.waits == [0.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", ["600", _http_date(600)], ids=["seconds", "http-date"])
async def test_a_retry_after_over_the_cap_waits_only_the_cap(retry_after) -> None:
    http, _requests = _http(httpx.Response(429, headers={"Retry-After": retry_after}), httpx.Response(200))

    async with http.client(BASE_URL) as client:
        await client.get("/items")

    assert http.waits == [60.0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retry_after",
    [None, "soon", "\u00b2", "\u0661\u0662"],
    ids=["absent", "unreadable", "superscript-digit", "arabic-indic-digits"],
)
async def test_without_a_readable_retry_after_the_wait_backs_off(retry_after) -> None:
    # Raw header bytes, as a server sends them; httpx decodes them as UTF-8.
    headers = [(b"Retry-After", retry_after.encode())] if retry_after else []
    http, requests = _http(httpx.Response(429, headers=headers))

    async with http.client(BASE_URL) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get("/items")

    assert len(requests) == 4
    assert http.waits == [2.0, 4.0, 6.0]


@pytest.mark.asyncio
async def test_a_status_that_asks_to_wait_is_retried_for_a_post_that_is_not_marked_safe() -> None:
    http, requests = _http(httpx.Response(503), httpx.Response(201))

    async with http.client(BASE_URL) as client:
        response = await client.post("/items", json={"name": "x"})

    assert response.status_code == 201
    assert len(requests) == 2


# ---------------------------------------------------------------------------
# Replaying other 5xx and transport errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "PUT", "DELETE"])
async def test_a_server_error_is_retried_for_an_idempotent_method(method) -> None:
    http, requests = _http(httpx.Response(502), httpx.Response(200))

    async with http.client(BASE_URL) as client:
        response = await client.request(method, "/items")

    assert response.status_code == 200
    assert len(requests) == 2
    assert http.waits == [2.0]


@pytest.mark.asyncio
async def test_a_server_error_is_retried_for_a_post_marked_as_a_safe_read() -> None:
    http, requests = _http(httpx.Response(500), httpx.Response(200, json={"issues": []}))

    async with http.client(BASE_URL) as client:
        response = await client.post("/search", json={"jql": "project = PAY"}, safe_read=True)

    assert response.json() == {"issues": []}
    assert [request.method for request in requests] == ["POST", "POST"]


@pytest.mark.asyncio
async def test_a_server_error_on_an_unmarked_post_is_returned_without_a_retry() -> None:
    http, requests = _http(httpx.Response(500), httpx.Response(200))

    async with http.client(BASE_URL) as client:
        response = await client.post("/items", json={"name": "x"})

    assert response.status_code == 500
    assert len(requests) == 1
    assert http.waits == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("All connection attempts failed"), httpx.ReadTimeout("timed out")],
    ids=["connect", "read-timeout"],
)
async def test_a_transport_error_is_retried_after_a_short_fixed_wait(error) -> None:
    http, requests = _http(error, httpx.Response(200))

    async with http.client(BASE_URL) as client:
        response = await client.get("/items")

    assert response.status_code == 200
    assert len(requests) == 2
    assert http.waits == [1.0]


@pytest.mark.asyncio
async def test_a_transport_error_that_persists_raises_unreachable_with_the_vpn_hint() -> None:
    http, requests = _http(httpx.ConnectError("All connection attempts failed"))

    async with http.client(BASE_URL, product_name="Jira") as client:
        with pytest.raises(ConnectorUnreachableError) as raised:
            await client.post("/rest/api/2/search", json={}, safe_read=True)

    assert len(requests) == 4
    assert http.waits == [1.0, 1.0, 1.0]
    assert str(raised.value) == (
        "Jira request failed to reach /rest/api/2/search after 4 attempts (All connection attempts failed). "
        "The instance may be slow or unreachable; check VPN or network, then retry."
    )
    assert isinstance(raised.value.__cause__, httpx.ConnectError)


@pytest.mark.asyncio
async def test_a_transport_error_on_an_unmarked_post_is_not_retried() -> None:
    http, requests = _http(httpx.ReadTimeout("timed out"), httpx.Response(200))

    async with http.client(BASE_URL) as client:
        with pytest.raises(ConnectorUnreachableError, match="after 1 attempt "):
            await client.post("/items", json={})

    assert len(requests) == 1


# ---------------------------------------------------------------------------
# Giving up, and what is returned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_exhausted_rate_limit_raises_with_the_last_response_and_rate_limit_wording() -> None:
    http, requests = _http(*(httpx.Response(429, text=f"attempt {number}") for number in range(1, 5)))

    async with http.client(BASE_URL, product_name="Confluence") as client:
        with pytest.raises(httpx.HTTPStatusError) as raised:
            await client.get("/wiki/rest/api/content/1")

    assert len(requests) == 4
    assert raised.value.response.text == "attempt 4"
    assert str(raised.value) == (
        "Confluence rate limit persisted after 4 attempts for /wiki/rest/api/content/1. "
        "Retry later or reduce the source sync frequency."
    )


@pytest.mark.asyncio
async def test_an_exhausted_server_error_raises_with_the_last_response() -> None:
    http, requests = _http(*(httpx.Response(503, text=f"attempt {number}") for number in range(1, 5)))

    async with http.client(BASE_URL, product_name="Jira") as client:
        with pytest.raises(httpx.HTTPStatusError) as raised:
            await client.get("/rest/api/2/issue/PAY-1")

    assert len(requests) == 4
    assert raised.value.response.status_code == 503
    assert raised.value.response.text == "attempt 4"
    assert "Jira answered 503 Service Unavailable after 4 attempts" in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 301, 400, 401, 403, 404, 410])
async def test_a_status_that_is_not_retried_is_returned_unraised(status) -> None:
    http, requests = _http(httpx.Response(status, headers={"Location": "/elsewhere"} if status == 301 else {}))

    async with http.client(BASE_URL, follow_redirects=False) as client:
        response = await client.get("/items")

    assert response.status_code == status
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_a_request_policy_replaces_the_clients_for_that_request() -> None:
    http, requests = _http(httpx.Response(503))

    async with http.client(BASE_URL) as client:
        with pytest.raises(httpx.HTTPStatusError, match="after 2 attempts"):
            await client.get("/items", retry=RetryPolicy(attempts=2, backoff_seconds=0.5))

    assert len(requests) == 2
    assert http.waits == [0.5]


@pytest.mark.asyncio
async def test_an_unreadable_retry_after_still_paces_the_limiter_and_closes_the_stream() -> None:
    clock = FakeClock()
    limiter = AtlassianRequestLimiter(min_interval_seconds=0.5, clock=clock)
    handle, requests = _answers(httpx.Response(503, headers=[(b"Retry-After", "\u00b2".encode())]), httpx.Response(200, content=b"ok"))
    client = ConnectorHttpClient(
        product_name="Confluence",
        base_url=BASE_URL,
        timeout=5.0,
        follow_redirects=True,
        limiter=limiter,
        sleep=clock.sleep,
        clock=clock,
        transport=httpx.MockTransport(handle),
    )

    async with client:
        async with client.stream("GET", "/download/1") as response:
            assert await response.aread() == b"ok"

    # Read as absent: the backoff, then the next attempt spaced only by the limiter's interval.
    assert clock.waits == [2.0]
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(("seconds_to_deadline", "retried"), [(6.0, False), (8.0, True)])
async def test_a_retry_is_made_only_when_its_wait_and_timeout_fit_before_the_deadline(
    seconds_to_deadline, retried
) -> None:
    clock = FakeClock()
    handle, requests = _answers(httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(200))
    client = ConnectorHttpClient(
        product_name="Test",
        base_url=BASE_URL,
        timeout=5.0,
        follow_redirects=True,
        sleep=clock.sleep,
        clock=clock,
        transport=httpx.MockTransport(handle),
    )

    async with client:
        deadline = clock() + seconds_to_deadline
        if retried:
            assert (await client.get("/items", deadline=deadline)).status_code == 200
        else:
            with pytest.raises(httpx.HTTPStatusError, match="after 1 attempt "):
                await client.get("/items", deadline=deadline)

    # A 2 s wait and a 5 s timeout need 7 s.
    assert len(requests) == (2 if retried else 1)


@pytest.mark.asyncio
async def test_a_wait_that_would_pass_the_deadline_gives_up_instead() -> None:
    http, requests = _http(httpx.Response(429, headers={"Retry-After": "30"}), httpx.Response(200))

    async with http.client(BASE_URL) as client:
        with pytest.raises(httpx.HTTPStatusError, match="after 1 attempt "):
            await client.get("/items", deadline=time.monotonic() + 10)

    assert len(requests) == 1
    assert http.waits == []


# ---------------------------------------------------------------------------
# fail_fast and the limiter
# ---------------------------------------------------------------------------


class QuotaIsZero(RuntimeError):
    pass


@pytest.mark.asyncio
async def test_fail_fast_sees_every_response_and_stops_the_loop() -> None:
    seen: list[int] = []

    def fail_fast(response: httpx.Response) -> None:
        seen.append(response.status_code)
        if response.headers.get("X-RateLimit-Limit") == "0":
            raise QuotaIsZero("quota is zero")

    http, requests = _http(
        httpx.Response(503),
        httpx.Response(429, headers={"X-RateLimit-Limit": "0", "Retry-After": "9223372036854775807"}),
        httpx.Response(200),
    )

    async with http.client(BASE_URL, fail_fast=fail_fast) as client:
        with pytest.raises(QuotaIsZero):
            await client.get("/items")

    assert seen == [503, 429]
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_every_attempt_passes_through_the_limiter_with_the_wait_it_asked_for() -> None:
    limiter = RecordingLimiter()
    http, requests = _http(
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(503),
        httpx.Response(500, headers={"Retry-After": "3"}),
        httpx.Response(200),
    )

    async with http.client(BASE_URL, limiter=limiter) as client:
        await client.get("/items")

    assert limiter.attempts == len(requests) == 4
    # A 429 spaces the limiter by its wait; a 5xx only when it carries Retry-After.
    assert limiter.delays == [7.0, None, 3.0, None]


@pytest.mark.asyncio
async def test_a_request_marked_unlimited_goes_past_the_limiter() -> None:
    limiter = RecordingLimiter()
    http, _requests = _http(httpx.Response(200))

    async with http.client(BASE_URL, limiter=limiter) as client:
        await client.get("/images/1.png", limited=False)
        await client.get("/items")

    assert limiter.attempts == 1


@pytest.mark.asyncio
async def test_a_wait_asked_for_spaces_the_origin_once_by_at_most_the_cap() -> None:
    clock = FakeClock()
    limiter = AtlassianRequestLimiter(min_interval_seconds=0.5, clock=clock)
    handle, requests = _answers(
        httpx.Response(503, headers={"Retry-After": "600"}), httpx.Response(200), httpx.Response(200)
    )
    client = ConnectorHttpClient(
        product_name="Jira",
        base_url=BASE_URL,
        timeout=5.0,
        follow_redirects=True,
        limiter=limiter,
        sleep=clock.sleep,
        transport=httpx.MockTransport(handle),
    )

    async with client:
        await client.get("/items")
        await client.get("/items")

    # The retry waits the capped 60 s once; the limiter, already past its hold, adds nothing.
    # The next request then keeps only the origin's 0.5 s interval.
    assert clock.waits == [60.0, 0.5]
    assert len(requests) == 3


@pytest.mark.asyncio
async def test_retry_after_on_a_server_error_that_is_not_retried_only_spaces_the_limiter() -> None:
    clock = FakeClock()
    limiter = AtlassianRequestLimiter(min_interval_seconds=0.5, clock=clock)
    handle, _requests = _answers(httpx.Response(500, headers={"Retry-After": "3"}), httpx.Response(200))
    client = ConnectorHttpClient(
        product_name="Jira",
        base_url=BASE_URL,
        timeout=5.0,
        follow_redirects=True,
        limiter=limiter,
        sleep=clock.sleep,
        transport=httpx.MockTransport(handle),
    )

    async with client:
        degraded = await client.post("/items", json={})
        await client.get("/items")

    assert degraded.status_code == 500
    assert clock.waits == [3.0]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class _BodyThatBreaks(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"first chunk"
        raise httpx.ReadError("connection reset")


@pytest.mark.asyncio
async def test_a_stream_is_retried_until_it_opens() -> None:
    http, requests = _http(httpx.Response(503), httpx.ConnectError("refused"), httpx.Response(200, content=b"body"))

    async with http.client(BASE_URL) as client:
        async with client.stream("GET", "/download/1") as response:
            body = b"".join([chunk async for chunk in response.aiter_bytes()])

    assert body == b"body"
    assert len(requests) == 3
    assert http.waits == [2.0, 1.0]


@pytest.mark.asyncio
async def test_a_stream_that_breaks_while_reading_the_body_is_not_retried() -> None:
    http, requests = _http(httpx.Response(200, stream=_BodyThatBreaks()), httpx.Response(200, content=b"again"))

    async with http.client(BASE_URL) as client:
        async with client.stream("GET", "/download/1") as response:
            with pytest.raises(httpx.ReadError):
                _ = [chunk async for chunk in response.aiter_bytes()]

    assert len(requests) == 1


@pytest.mark.asyncio
async def test_an_exhausted_stream_raises_with_the_last_response_body_read() -> None:
    http, requests = _http(httpx.Response(503, content=b"busy"))

    async with http.client(BASE_URL) as client:
        with pytest.raises(httpx.HTTPStatusError) as raised:
            async with client.stream("GET", "/download/1"):
                pytest.fail("an exhausted stream never opens")

    assert len(requests) == 4
    assert raised.value.response.text == "busy"
    assert raised.value.response.is_closed


@pytest.mark.asyncio
async def test_a_stream_returns_a_status_that_is_not_retried_for_the_caller_to_raise() -> None:
    http, requests = _http(httpx.Response(404, content=b"missing"))

    async with http.client(BASE_URL) as client:
        async with client.stream("GET", "/download/1") as response:
            assert response.status_code == 404
            with pytest.raises(httpx.HTTPStatusError):
                response.raise_for_status()

    assert len(requests) == 1


# ---------------------------------------------------------------------------
# Transport settings and lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_client_sends_its_base_url_and_headers_and_follows_redirects_only_when_asked() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(302, headers={"Location": f"{BASE_URL}/new"})
        return httpx.Response(200, json={"authorization": request.headers["Authorization"]})

    http = FakeConnectorHttp(handle)
    following = http.client(BASE_URL, headers={"Authorization": "Bearer token"}, follow_redirects=True)
    staying = http.client(BASE_URL, follow_redirects=False)
    async with following, staying:
        assert (await following.get("/old")).json() == {"authorization": "Bearer token"}
        assert (await staying.get("/old")).status_code == 302


@pytest.mark.asyncio
async def test_aclose_closes_the_client() -> None:
    http, _requests = _http(httpx.Response(200))
    client = http.client(BASE_URL)

    assert not client.is_closed
    await client.aclose()
    assert client.is_closed
    await client.aclose()
    assert client.is_closed
