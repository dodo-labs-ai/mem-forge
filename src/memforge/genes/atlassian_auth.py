"""Atlassian rules shared by the Confluence and Jira genes.

The genes send their requests through
:class:`memforge.genes.toolkit.http.ConnectorHttpClient`. This module holds
what is Atlassian's own:

- :func:`atlassian_request_limiter`: the process-wide pace for one Atlassian
  origin, passed to the client as its ``limiter``;
- :func:`atlassian_fail_fast`: the client's ``fail_fast`` hook, which stops
  at once when the credential has no REST quota;
- :func:`atlassian_record_state`: what reading one record by id says about
  its absence.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from threading import Lock
from urllib.parse import urlsplit

import httpx

from memforge.genes.toolkit.http import Sleep
from memforge.models import AbsenceAnswer

# Responses that say the requested record does not exist (any longer).
ATLASSIAN_ABSENT_STATUS_CODES = frozenset({httpx.codes.NOT_FOUND, httpx.codes.GONE})
# A refusal to show one record the credential otherwise reads: the record may
# still exist, so it is never taken as absence.
ATLASSIAN_RECORD_FORBIDDEN_STATUS_CODE = httpx.codes.FORBIDDEN
_ATLASSIAN_LIMITERS_LOCK = Lock()
_ATLASSIAN_REQUEST_LIMITERS: dict[str, "AtlassianRequestLimiter"] = {}


class AtlassianZeroQuotaRateLimitError(RuntimeError):
    """Raised when an Atlassian REST endpoint reports no usable API quota."""


class AtlassianRequestLimiter:
    """Serializes Atlassian REST attempts for one origin and enforces a minimum gap between them.

    Each Source reading the origin registers its interval as an owner, and
    the strictest interval wins. One attempt runs at a time; the next starts
    no sooner than the interval after the previous one ended, or later when
    that response asked to wait (``delay_for_response``)::

        limiter = atlassian_request_limiter(base_url, min_interval_seconds=0.75, owner_id=source_id)
        client = ConnectorHttpClient(..., limiter=limiter)

    ``clock`` reads the time, ``time.monotonic`` by default, and the wait
    uses the ``sleep`` each caller passes to :meth:`run`; tests replace both.
    """

    def __init__(self, *, min_interval_seconds: float = 0.0, clock: Callable[[], float] | None = None) -> None:
        self._min_interval_seconds = max(min_interval_seconds, 0.0)
        self._owner_intervals: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._next_request_at = 0.0
        self._clock = clock

    def configure(self, *, min_interval_seconds: float, owner_id: str | None = None) -> None:
        interval = max(min_interval_seconds, 0.0)
        if owner_id:
            self._owner_intervals[owner_id] = interval
            self._min_interval_seconds = max(self._owner_intervals.values(), default=interval)
            return
        self._min_interval_seconds = max(self._min_interval_seconds, interval)

    def release(self, owner_id: str) -> None:
        self._owner_intervals.pop(owner_id, None)
        self._min_interval_seconds = max(self._owner_intervals.values(), default=0.0)
        if not self._owner_intervals:
            self._next_request_at = 0.0

    @property
    def has_owners(self) -> bool:
        return bool(self._owner_intervals)

    async def run(
        self,
        operation: Callable[[], Awaitable[httpx.Response]],
        *,
        delay_for_response: Callable[[httpx.Response], float | None] | None = None,
        sleep: Sleep | None = None,
    ) -> httpx.Response:
        async with self._lock:
            now = self._now()
            if now < self._next_request_at:
                await (sleep or asyncio.sleep)(self._next_request_at - now)
            response: httpx.Response | None = None
            try:
                response = await operation()
                return response
            finally:
                response_delay = delay_for_response(response) if response is not None and delay_for_response else None
                now = self._now()
                self._next_request_at = now + self._min_interval_seconds
                if response_delay is not None:
                    self._next_request_at = max(self._next_request_at, now + max(response_delay, 0.0))

    def _now(self) -> float:
        return self._clock() if self._clock is not None else time.monotonic()


async def atlassian_record_state(read: Awaitable[httpx.Response]) -> AbsenceAnswer:
    """What reading one record by id says about it: absent, exists or hidden.

    Not found or gone (404, 410) is ``absent``, a permission refusal for this
    one record (403) is ``hidden``, and a successful read is ``exists``. Any
    other error raises, so the run proves nothing from it: an expired
    credential (401), a rate limit that outlasted its retries (429) or a
    server error (5xx).
    """

    try:
        await read
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in ATLASSIAN_ABSENT_STATUS_CODES:
            return AbsenceAnswer.ABSENT
        if status == ATLASSIAN_RECORD_FORBIDDEN_STATUS_CODE:
            return AbsenceAnswer.HIDDEN
        raise
    return AbsenceAnswer.EXISTS


def atlassian_request_limiter(
    base_url: str,
    *,
    min_interval_seconds: float,
    owner_id: str | None = None,
) -> AtlassianRequestLimiter:
    """Return the process-wide request limiter for an Atlassian origin."""
    parts = urlsplit(base_url)
    key = f"{parts.scheme.lower()}://{parts.netloc.lower()}"
    with _ATLASSIAN_LIMITERS_LOCK:
        limiter = _ATLASSIAN_REQUEST_LIMITERS.get(key)
        if limiter is None:
            limiter = AtlassianRequestLimiter()
            _ATLASSIAN_REQUEST_LIMITERS[key] = limiter
        limiter.configure(min_interval_seconds=min_interval_seconds, owner_id=owner_id)
        return limiter


def release_atlassian_request_limiter(base_url: str, *, owner_id: str) -> None:
    """Release one owner's contribution to the limiter for an Atlassian origin."""
    parts = urlsplit(base_url)
    key = f"{parts.scheme.lower()}://{parts.netloc.lower()}"
    with _ATLASSIAN_LIMITERS_LOCK:
        limiter = _ATLASSIAN_REQUEST_LIMITERS.get(key)
        if limiter is not None:
            limiter.release(owner_id)
            if not limiter.has_owners:
                _ATLASSIAN_REQUEST_LIMITERS.pop(key, None)


def atlassian_fail_fast(product_name: str, zero_quota_message: str | None = None) -> Callable[[httpx.Response], None]:
    """The ``fail_fast`` hook for an Atlassian ``ConnectorHttpClient``: stop at once when the REST quota is zero.

    Atlassian answers 429 with ``X-RateLimit-Limit: 0`` and
    ``X-RateLimit-FillRate: 0`` when the credential has no API quota at all;
    waiting never helps, so the hook raises
    :class:`AtlassianZeroQuotaRateLimitError` with ``zero_quota_message``, or
    a message naming ``product_name``::

        ConnectorHttpClient(..., fail_fast=atlassian_fail_fast("Jira", "Jira PAT API quota is zero ..."))
    """
    message = zero_quota_message or (
        f"{product_name} REST API quota is zero for this request. "
        f"Ask the {product_name} administrator to enable REST API quota."
    )

    def fail_fast(response: httpx.Response) -> None:
        if _is_zero_quota_rate_limit(response):
            raise AtlassianZeroQuotaRateLimitError(message)

    return fail_fast


def _is_zero_quota_rate_limit(resp: httpx.Response) -> bool:
    limit = str(resp.headers.get("X-RateLimit-Limit") or "").strip()
    fillrate = str(resp.headers.get("X-RateLimit-FillRate") or "").strip()
    return resp.status_code == httpx.codes.TOO_MANY_REQUESTS and limit == "0" and fillrate == "0"
