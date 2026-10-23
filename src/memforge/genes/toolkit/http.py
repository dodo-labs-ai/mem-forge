"""Send a connector's HTTP requests with one retry and pacing policy.

:class:`ConnectorHttpClient` wraps ``httpx.AsyncClient``. A connector builds
one in ``authenticate``, keeps it in ``self._client`` so that ``Gene.close()``
releases it, and sends every provider request through it::

    from memforge.genes.toolkit.config import bearer_headers, tls_verify
    from memforge.genes.toolkit.http import ConnectorHttpClient

    async def authenticate(self) -> None:
        await self.close()
        self._client = ConnectorHttpClient(
            product_name="Wiki",
            base_url=self.config["base_url"],
            headers=bearer_headers(self.config, "Wiki"),
            timeout=30.0,
            verify=tls_verify(self.config),
            follow_redirects=True,
        )

    async def _page(self, page_id: str) -> dict:
        response = await self._client.get(f"/api/pages/{page_id}")
        response.raise_for_status()
        return response.json()

The client decides when to send a request again; the connector decides what
an answer means:

- **Wait and retry.** 429 and 503 are retried for every method, after the
  wait ``Retry-After`` asks for (seconds or an HTTP date). Without the header
  the wait grows with each attempt.
- **Replay.** Any other 5xx, and a transport error such as a refused
  connection or a timeout, is retried only for a request that is safe to send
  twice: ``GET``, ``HEAD``, ``OPTIONS``, ``PUT`` and ``DELETE``, or one the
  connector marks ``safe_read=True``, such as a search sent as ``POST``::

      response = await self._client.post("/rest/api/2/search", json=query, safe_read=True)

- **Give up.** When the attempts run out on a retryable status, the client
  raises ``httpx.HTTPStatusError`` carrying the last response. When they run
  out on a transport error, it raises :class:`ConnectorUnreachableError`. A
  listing that meets either fails, so a retry never weakens its completeness.
- **Everything else** is returned as it arrived, including 4xx, and the
  connector raises with ``response.raise_for_status()`` or reads the status
  itself, as Jira does to word an expired browser session.

:class:`RetryPolicy` holds the numbers; every wait is capped at
``max_wait_seconds``, ``Retry-After`` included.

A body is streamed with :meth:`ConnectorHttpClient.stream`. Only opening the
stream is retried; once the context is entered the response is the
connector's, and a failure while reading it raises::

    async with self._client.stream("GET", download_path) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            ...

Two hooks fit a provider's own rules, and both see every attempt:

- ``fail_fast(response)`` runs on each response before the client decides to
  retry, and raises to stop at once, as Atlassian does when it reports a REST
  quota of zero.
- ``limiter`` paces the requests that share it (see :class:`RequestLimiter`).
  Every attempt, retries included, passes through it, and a response that
  asks to wait spaces the limiter's next request by the same capped wait.

Tests pass ``transport=httpx.MockTransport(handler)`` and a ``sleep`` that
returns at once; the client and its limiter wait only through ``sleep``.
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import httpx

__all__ = [
    "DEFAULT_RETRY_POLICY",
    "IDEMPOTENT_METHODS",
    "ConnectorHttpClient",
    "ConnectorUnreachableError",
    "RequestLimiter",
    "RetryPolicy",
    "Sleep",
]

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]

# Methods a server may receive twice without a different outcome (RFC 9110 §9.2.2).
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
# "Not now": the server did not act on the request, so any method is sent again.
_WAIT_STATUSES = frozenset({httpx.codes.TOO_MANY_REQUESTS, httpx.codes.SERVICE_UNAVAILABLE})


@dataclass(frozen=True)
class RetryPolicy:
    """How often a request is sent, and how long the client waits between attempts.

    The defaults suit a provider API read during a sync::

        RetryPolicy()  # 4 attempts; waits of 2, 4 and 6 s without Retry-After; at most 60 s each

    - ``attempts``: how many times one request is sent in all, the first
      included.
    - ``backoff_seconds``: without ``Retry-After``, the wait before attempt
      *n + 1* is ``backoff_seconds × n``.
    - ``max_wait_seconds``: no wait is longer, whatever ``Retry-After`` asks
      for, so a provider can stall a run, or every request sharing a limiter,
      for at most this long per response.
    - ``transport_wait_seconds``: the fixed wait after a transport error. A
      flaky connection (a cold start, a dual-stack IPv6 race, VPN or DNS
      warm-up) usually succeeds on the next try, so it waits briefly instead
      of backing off.
    """

    attempts: int = 4
    backoff_seconds: float = 2.0
    max_wait_seconds: float = 60.0
    transport_wait_seconds: float = 1.0

    def wait_after(self, response: httpx.Response, attempt: int) -> float:
        """The wait after ``response`` answered attempt ``attempt``: its ``Retry-After``, or the backoff, capped::

            RetryPolicy().wait_after(httpx.Response(429, headers={"Retry-After": "5"}), 1)    # 5.0
            RetryPolicy().wait_after(httpx.Response(503, headers={"Retry-After": "600"}), 1)  # 60.0
            RetryPolicy().wait_after(httpx.Response(500), 2)                                  # 4.0
        """
        requested = _retry_after_seconds(response)
        wait = requested if requested is not None else self.backoff_seconds * attempt
        return min(max(wait, 0.0), self.max_wait_seconds)


DEFAULT_RETRY_POLICY = RetryPolicy()


class RequestLimiter(Protocol):
    """Paces the requests of every client that shares it, such as all Sources reading one Atlassian origin.

    ``run`` sends one attempt through ``operation`` when the pace allows,
    waiting with ``sleep`` until then. After the attempt, a non-``None``
    ``delay_for_response(response)`` holds the limiter's next request back by
    that many seconds. ``memforge.genes.atlassian_auth.AtlassianRequestLimiter``
    is the built-in one.
    """

    async def run(
        self,
        operation: Callable[[], Awaitable[httpx.Response]],
        *,
        delay_for_response: Callable[[httpx.Response], float | None] | None = None,
        sleep: Sleep | None = None,
    ) -> httpx.Response: ...


class ConnectorUnreachableError(httpx.TransportError):
    """A request kept failing to reach the provider until its attempts ran out.

    The message names the product, the URL, the number of attempts and the
    last transport error, and asks the user to check the VPN or network.
    """


class ConnectorHttpClient:
    """An ``httpx.AsyncClient`` that retries and paces a connector's requests; see the module docstring.

    Every transport setting is the connector's explicit choice:

    - ``product_name`` words the errors, such as "Confluence".
    - ``base_url`` and ``headers`` apply to every request.
    - ``timeout`` is a number of seconds or an ``httpx.Timeout``.
    - ``verify`` is ``True`` or the ``ssl.SSLContext`` that
      :func:`memforge.genes.toolkit.config.tls_verify` returns.
    - ``follow_redirects`` says whether a 3xx is followed.

    ``limiter``, ``fail_fast`` and ``retry`` set the policy (module
    docstring). ``sleep`` and ``transport`` exist for tests.

    Close it with ``await client.aclose()``; ``is_closed`` tells whether it
    is. ``Gene.close()`` does this for the client in ``self._client``.
    """

    def __init__(
        self,
        *,
        product_name: str,
        base_url: str = "",
        headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout | float,
        verify: ssl.SSLContext | bool = True,
        follow_redirects: bool,
        limiter: RequestLimiter | None = None,
        fail_fast: Callable[[httpx.Response], None] | None = None,
        retry: RetryPolicy = DEFAULT_RETRY_POLICY,
        sleep: Sleep = asyncio.sleep,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.product_name = product_name
        self._limiter = limiter
        self._fail_fast = fail_fast
        self._retry = retry
        self._sleep = sleep
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers=dict(headers or {}),
            timeout=timeout,
            verify=verify,
            follow_redirects=follow_redirects,
            transport=transport,
        )

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout | float | Any = httpx.USE_CLIENT_DEFAULT,
        safe_read: bool = False,
        limited: bool = True,
        retry: RetryPolicy | None = None,
        deadline: float | None = None,
    ) -> httpx.Response:
        """Send one request, retrying as the module docstring describes, and return the response with its body read.

        - ``safe_read=True`` marks a request that only reads although its
          method is not idempotent, so that a 5xx or a transport error is
          retried as for ``GET``.
        - ``limited=False`` sends it past the client's limiter.
        - ``retry`` replaces the client's :class:`RetryPolicy` for this
          request.
        - ``deadline`` is a ``time.monotonic()`` instant no retry wait may end
          after; at the deadline the client gives up as if the attempts had
          run out.

        ``params``, ``json``, ``headers`` and ``timeout`` are passed to httpx.
        """

        async def send() -> httpx.Response:
            return await self._http.request(method, url, params=params, json=json, headers=headers, timeout=timeout)

        return await self._send_with_retries(
            send,
            method=method,
            url=url,
            streamed=False,
            safe_read=safe_read,
            limited=limited,
            policy=retry or self._retry,
            deadline=deadline,
        )

    async def get(self, url: str, **options: Any) -> httpx.Response:
        """``request("GET", url, ...)``."""
        return await self.request("GET", url, **options)

    async def post(self, url: str, **options: Any) -> httpx.Response:
        """``request("POST", url, ...)``; pass ``safe_read=True`` for a ``POST`` that only reads."""
        return await self.request("POST", url, **options)

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout | float | Any = httpx.USE_CLIENT_DEFAULT,
        safe_read: bool = False,
        limited: bool = True,
        retry: RetryPolicy | None = None,
        deadline: float | None = None,
    ) -> AsyncIterator[httpx.Response]:
        """Open a response without reading its body; the options are those of :meth:`request`.

        Opening is retried until the headers of a response the client keeps
        arrive. The body is never retried: an error while reading it raises
        from ``aiter_bytes``. The response closes when the block exits.
        """

        async def send() -> httpx.Response:
            built = self._http.build_request(method, url, params=params, headers=headers, timeout=timeout)
            return await self._http.send(built, stream=True)

        response = await self._send_with_retries(
            send,
            method=method,
            url=url,
            streamed=True,
            safe_read=safe_read,
            limited=limited,
            policy=retry or self._retry,
            deadline=deadline,
        )
        try:
            yield response
        finally:
            await response.aclose()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_closed(self) -> bool:
        return self._http.is_closed

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> ConnectorHttpClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Retry loop
    # ------------------------------------------------------------------

    async def _send_with_retries(
        self,
        send: Callable[[], Awaitable[httpx.Response]],
        *,
        method: str,
        url: str,
        streamed: bool,
        safe_read: bool,
        limited: bool,
        policy: RetryPolicy,
        deadline: float | None,
    ) -> httpx.Response:
        replayable = safe_read or method.upper() in IDEMPOTENT_METHODS
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._attempt(send, policy=policy, attempt=attempt, limited=limited)
            except httpx.TransportError as exc:
                wait = policy.transport_wait_seconds
                if not replayable or attempt >= policy.attempts or _passes(deadline, wait):
                    raise self._unreachable(url, attempt, exc) from exc
                logger.warning(
                    "%s request to %s failed in transport (%s); retrying in %.1f s",
                    self.product_name,
                    url,
                    type(exc).__name__,
                    wait,
                )
                await self._sleep(wait)
                continue

            try:
                if self._fail_fast is not None:
                    self._fail_fast(response)
            except BaseException:
                await response.aclose()
                raise
            status = response.status_code
            if not (status in _WAIT_STATUSES or (replayable and httpx.codes.is_server_error(status))):
                return response

            wait = policy.wait_after(response, attempt)
            if attempt >= policy.attempts or _passes(deadline, wait):
                if streamed:
                    await _read_and_close(response)
                raise self._exhausted(response, url, attempt)
            logger.warning(
                "%s answered %d for %s; retrying in %.1f s (attempt %d of %d)",
                self.product_name,
                status,
                url,
                wait,
                attempt + 1,
                policy.attempts,
            )
            if streamed:
                await response.aclose()
            await self._sleep(wait)

    async def _attempt(
        self,
        send: Callable[[], Awaitable[httpx.Response]],
        *,
        policy: RetryPolicy,
        attempt: int,
        limited: bool,
    ) -> httpx.Response:
        if not limited or self._limiter is None:
            return await send()
        return await self._limiter.run(
            send,
            delay_for_response=lambda response: _limiter_delay(policy, response, attempt),
            sleep=self._sleep,
        )

    def _exhausted(self, response: httpx.Response, url: str, attempts: int) -> httpx.HTTPStatusError:
        tried = _attempts_phrase(attempts)
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            message = (
                f"{self.product_name} rate limit persisted after {tried} for {url}. "
                "Retry later or reduce the source sync frequency."
            )
        else:
            message = (
                f"{self.product_name} answered {response.status_code} {response.reason_phrase} "
                f"after {tried} for {url}. Retry later."
            )
        return httpx.HTTPStatusError(message, request=response.request, response=response)

    def _unreachable(self, url: str, attempts: int, exc: httpx.TransportError) -> ConnectorUnreachableError:
        detail = str(exc) or type(exc).__name__
        try:
            request = exc.request
        except RuntimeError:
            request = None
        return ConnectorUnreachableError(
            f"{self.product_name} request failed to reach {url} after {_attempts_phrase(attempts)} "
            f"({detail}). The instance may be slow or unreachable; check VPN or network, then retry.",
            request=request,
        )


def _limiter_delay(policy: RetryPolicy, response: httpx.Response, attempt: int) -> float | None:
    """How long a response holds back the next request through the limiter, or ``None``.

    A 429 always does, by the wait the client itself takes. A 5xx does only
    when it carries ``Retry-After``, whether or not the client retries it.
    """
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        return policy.wait_after(response, attempt)
    if httpx.codes.is_server_error(response.status_code) and response.headers.get("Retry-After", "").strip():
        return policy.wait_after(response, attempt)
    return None


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The wait ``Retry-After`` asks for, in seconds or as an HTTP date, or ``None`` when absent or unreadable."""
    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0)


def _passes(deadline: float | None, wait: float) -> bool:
    return deadline is not None and time.monotonic() + wait > deadline


def _attempts_phrase(attempts: int) -> str:
    return "1 attempt" if attempts == 1 else f"{attempts} attempts"


async def _read_and_close(response: httpx.Response) -> None:
    """Read a streamed error response, so the raised error carries its body, then close it."""
    try:
        await response.aread()
    except httpx.HTTPError:
        pass
    finally:
        await response.aclose()
