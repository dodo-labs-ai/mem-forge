"""The connector toolkit's request rate: spaced starts, one attempt at a time, one limiter per key."""

from __future__ import annotations

import asyncio
import gc
from weakref import WeakValueDictionary

import httpx
import pytest

from memforge.genes.toolkit import pacing
from memforge.genes.toolkit.http import ConnectorHttpClient
from memforge.genes.toolkit.pacing import RateLimiter, shared_rate_limiter

BASE_URL = "https://provider.example.test"


class FakeClock:
    """A monotonic clock that advances only when something sleeps on it or a test moves it on."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def own_shared_limiters(monkeypatch):
    """Give each test its own registry of shared limiters."""
    monkeypatch.setattr(pacing, "_SHARED_LIMITERS", WeakValueDictionary())


def _answer_at(clock: FakeClock, starts: list[float], *, took: float = 0.0, status: int = 200):
    """An operation that records when it started, takes ``took`` seconds of the clock, and answers ``status``."""

    async def operation() -> httpx.Response:
        starts.append(clock.now)
        clock.now += took
        return httpx.Response(status)

    return operation


@pytest.mark.asyncio
async def test_a_request_starts_no_sooner_than_the_interval_after_the_previous_start() -> None:
    clock = FakeClock()
    limiter = RateLimiter(60, clock=clock)
    starts: list[float] = []

    # The first request takes 0.4 s, so the second waits only the rest of the 1 s interval.
    await limiter.run(_answer_at(clock, starts, took=0.4), sleep=clock.sleep)
    # The second takes longer than the interval, so the third starts as soon as it ends.
    await limiter.run(_answer_at(clock, starts, took=1.5), sleep=clock.sleep)
    await limiter.run(_answer_at(clock, starts), sleep=clock.sleep)

    assert starts == [1000.0, 1001.0, 1002.5]
    assert clock.waits == [pytest.approx(0.6)]


@pytest.mark.asyncio
async def test_one_attempt_runs_at_a_time() -> None:
    clock = FakeClock()
    limiter = RateLimiter(600, clock=clock)
    running = 0
    most_running = 0

    async def operation() -> httpx.Response:
        nonlocal running, most_running
        running += 1
        most_running = max(most_running, running)
        for _ in range(3):
            await asyncio.sleep(0)
        running -= 1
        return httpx.Response(200)

    await asyncio.gather(*(limiter.run(operation, sleep=clock.sleep) for _ in range(5)))

    assert most_running == 1
    assert clock.waits == [pytest.approx(0.1)] * 4


@pytest.mark.asyncio
async def test_a_response_that_asks_to_wait_holds_the_next_request_back() -> None:
    clock = FakeClock()
    limiter = RateLimiter(180, clock=clock)
    starts: list[float] = []

    def asks_to_wait(response: httpx.Response) -> float | None:
        return 30.0 if response.status_code == 429 else None

    rate_limited = _answer_at(clock, starts, took=2.0, status=429)
    await limiter.run(rate_limited, delay_for_response=asks_to_wait, sleep=clock.sleep)
    await limiter.run(_answer_at(clock, starts), delay_for_response=asks_to_wait, sleep=clock.sleep)

    # The wait counts from the end of the response that asked for it.
    assert starts == [1000.0, 1032.0]
    assert clock.waits == [30.0]


def test_clients_with_one_key_share_one_limiter() -> None:
    first = shared_rate_limiter("notion:3f2a9c", requests_per_minute=180)
    second = shared_rate_limiter("notion:3f2a9c", requests_per_minute=180)
    other = shared_rate_limiter("notion:77b0e1", requests_per_minute=180)

    assert first is second
    assert other is not first


def test_the_strictest_rate_for_one_key_wins() -> None:
    limiter = shared_rate_limiter("notion:3f2a9c", requests_per_minute=180)

    shared_rate_limiter("notion:3f2a9c", requests_per_minute=60)
    shared_rate_limiter("notion:3f2a9c", requests_per_minute=600)

    assert limiter.interval_seconds == 1.0


def test_a_limiter_no_client_holds_is_released() -> None:
    limiter = shared_rate_limiter("notion:3f2a9c", requests_per_minute=60)
    assert limiter.interval_seconds == 1.0

    del limiter
    gc.collect()

    # A fresh limiter: the strict rate asked for before is gone with the old one.
    assert shared_rate_limiter("notion:3f2a9c", requests_per_minute=180).interval_seconds == pytest.approx(1 / 3)


@pytest.mark.parametrize("rate", [0, -60])
def test_a_rate_that_is_not_positive_is_refused(rate) -> None:
    with pytest.raises(ValueError, match="requests_per_minute"):
        RateLimiter(rate)


@pytest.mark.asyncio
async def test_every_retry_passes_through_the_limiter() -> None:
    clock = FakeClock()
    starts: list[float] = []
    statuses = [503, 503, 200]

    def handle(request: httpx.Request) -> httpx.Response:
        starts.append(clock.now)
        return httpx.Response(statuses.pop(0))

    client = ConnectorHttpClient(
        product_name="Provider",
        base_url=BASE_URL,
        timeout=5.0,
        follow_redirects=False,
        limiter=RateLimiter(6, clock=clock),
        sleep=clock.sleep,
        transport=httpx.MockTransport(handle),
    )

    async with client:
        response = await client.get("/items")

    assert response.status_code == 200
    # The client backs off 2 s and then 4 s; the limiter holds each retry to the 10 s interval.
    assert starts == [1000.0, 1010.0, 1020.0]
    assert clock.waits == [2.0, 8.0, 4.0, 6.0]
