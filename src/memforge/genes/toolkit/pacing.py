"""Hold a connector to the request rate its provider allows.

A provider that grants a budget of requests per minute, such as Notion's
average of three requests per second per connection, is read through a
:class:`RateLimiter`. Every client that reads with the same credential shares
one limiter, so several Sources, or several runs of one Source, stay inside
the one budget the provider counts. A connector asks for that shared limiter
by a key naming the credential, never holding the secret itself, and passes
it to its client::

    from hashlib import sha256

    from memforge.genes.toolkit.http import ConnectorHttpClient
    from memforge.genes.toolkit.pacing import shared_rate_limiter

    token_key = "notion:" + sha256(token.encode()).hexdigest()[:16]
    self._client = ConnectorHttpClient(
        ...,
        limiter=shared_rate_limiter(token_key, requests_per_minute=180),
    )

The client sends every attempt through the limiter, retries included, and a
response that asks to wait (a 429, or a 5xx with ``Retry-After``) holds back
the next request of every client sharing it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from threading import Lock
from weakref import WeakValueDictionary

import httpx

from memforge.genes.toolkit.http import Sleep

__all__ = ["RateLimiter", "shared_rate_limiter"]

_SHARED_LIMITERS_LOCK = Lock()
_SHARED_LIMITERS: WeakValueDictionary[str, RateLimiter] = WeakValueDictionary()


class RateLimiter:
    """Starts at most ``requests_per_minute`` requests a minute, one at a time.

    It implements :class:`memforge.genes.toolkit.http.RequestLimiter`. One
    attempt runs at a time, and the next starts no sooner than
    ``60 / requests_per_minute`` seconds after the previous one started. At
    180 a minute, requests start a third of a second apart::

        limiter = RateLimiter(180)
        client = ConnectorHttpClient(..., limiter=limiter)

    A response that asks to wait pushes the next start back to the end of
    that wait, when that is later. The limiter reads the time through
    ``clock`` (``time.monotonic`` by default) and waits only through the
    ``sleep`` the client passes to :meth:`run`, so a test controls both.

    It spaces requests start to start, because a per-minute budget counts
    starts: a request that takes longer than the interval leaves the next one
    free to start as soon as it ends, and the budget is used in full.
    :class:`memforge.genes.atlassian_auth.AtlassianRequestLimiter` spaces end
    to start instead, keeping a quiet gap after every request to an Atlassian
    origin.
    """

    def __init__(self, requests_per_minute: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._interval_seconds = _interval_seconds(requests_per_minute)
        self._clock = clock
        self._lock = asyncio.Lock()
        self._next_start_at: float | None = None

    @property
    def interval_seconds(self) -> float:
        """The shortest time between two starts, in seconds: 60 divided by the rate."""
        return self._interval_seconds

    def tighten(self, requests_per_minute: float) -> None:
        """Lower the rate to ``requests_per_minute`` if that is stricter; a looser rate changes nothing::

            limiter = RateLimiter(180)
            limiter.tighten(60)   # now one request a second
            limiter.tighten(600)  # still one request a second
        """
        self._interval_seconds = max(self._interval_seconds, _interval_seconds(requests_per_minute))

    async def run(
        self,
        operation: Callable[[], Awaitable[httpx.Response]],
        *,
        delay_for_response: Callable[[httpx.Response], float | None] | None = None,
        sleep: Sleep | None = None,
    ) -> httpx.Response:
        """Send one attempt through ``operation`` when the rate allows, and return its response.

        After the attempt, a non-``None`` ``delay_for_response(response)``
        keeps the next attempt from starting until that many seconds after
        this one ended.
        """
        async with self._lock:
            now = self._clock()
            if self._next_start_at is not None and now < self._next_start_at:
                await (sleep or asyncio.sleep)(self._next_start_at - now)
            started_at = self._clock()
            response: httpx.Response | None = None
            try:
                response = await operation()
                return response
            finally:
                next_start_at = started_at + self._interval_seconds
                delay = delay_for_response(response) if response is not None and delay_for_response else None
                if delay is not None:
                    next_start_at = max(next_start_at, self._clock() + max(delay, 0.0))
                self._next_start_at = next_start_at


def shared_rate_limiter(key: str, *, requests_per_minute: float) -> RateLimiter:
    """The process-wide :class:`RateLimiter` for ``key``, made on first use.

    Every client built with the same key shares one limiter, so their
    requests together stay inside the rate. The key names what the provider
    counts against, such as a hash of the connection token; it never holds
    the secret itself::

        first = shared_rate_limiter("notion:3f2a9c", requests_per_minute=180)
        second = shared_rate_limiter("notion:3f2a9c", requests_per_minute=180)
        assert first is second

    When callers ask one key for different rates, the strictest rate any of
    them asked for while the limiter lives wins. The limiter lives as long
    as some client holds it; once none does, it is released, and the next
    call for the key starts a fresh limiter at the rate it asks for.
    """
    with _SHARED_LIMITERS_LOCK:
        limiter = _SHARED_LIMITERS.get(key)
        if limiter is None:
            limiter = RateLimiter(requests_per_minute)
            _SHARED_LIMITERS[key] = limiter
        else:
            limiter.tighten(requests_per_minute)
        return limiter


def _interval_seconds(requests_per_minute: float) -> float:
    if not requests_per_minute > 0:
        raise ValueError("requests_per_minute must be a positive number")
    return 60.0 / requests_per_minute
