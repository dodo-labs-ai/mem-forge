"""Shared authentication and request helpers for Atlassian-backed genes."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import httpx


logger = logging.getLogger(__name__)

ATLASSIAN_REQUEST_ATTEMPTS = 4
ATLASSIAN_DEFAULT_RETRY_DELAY_SECONDS = 2.0
ATLASSIAN_MAX_RETRY_DELAY_SECONDS = 60.0
# A flaky connection (cold start, dual-stack IPv6 race, VPN or DNS warmup) often
# succeeds on the next try, so a transient transport failure is retried after a
# short, fixed pause rather than the rate-limit backoff.
ATLASSIAN_TRANSPORT_RETRY_DELAY_SECONDS = 1.0
_ATLASSIAN_LIMITERS_LOCK = Lock()
_ATLASSIAN_REQUEST_LIMITERS: dict[str, "AtlassianRequestLimiter"] = {}


class AtlassianRateLimitError(RuntimeError):
    """Raised when an Atlassian REST endpoint remains rate-limited after retries."""


class AtlassianZeroQuotaRateLimitError(AtlassianRateLimitError):
    """Raised when an Atlassian REST endpoint reports no usable API quota."""


class AtlassianRequestTransportError(RuntimeError):
    """Raised when an Atlassian request keeps failing to connect after bounded retries."""


class AtlassianRequestLimiter:
    """Serializes Atlassian REST attempts for one origin and enforces a minimum gap."""

    def __init__(self, *, min_interval_seconds: float = 0.0) -> None:
        self._min_interval_seconds = max(min_interval_seconds, 0.0)
        self._owner_intervals: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._next_request_at = 0.0

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
    ) -> httpx.Response:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_request_at:
                await asyncio.sleep(self._next_request_at - now)
            response: httpx.Response | None = None
            try:
                response = await operation()
                return response
            finally:
                response_delay = delay_for_response(response) if response is not None and delay_for_response else None
                now = time.monotonic()
                self._next_request_at = now + self._min_interval_seconds
                if response_delay is not None:
                    self._next_request_at = max(self._next_request_at, now + max(response_delay, 0.0))


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


def resolve_pat(config: dict, product_name: str) -> str:
    """Return a PAT from config or an optional environment-variable reference."""
    pat = str(config.get("pat") or "").strip()
    if pat:
        return pat

    env_var = str(config.get("pat_env_var") or "").strip()
    if env_var:
        env_pat = os.environ.get(env_var, "").strip()
        if env_pat:
            return env_pat

    raise ValueError(f"{product_name} PAT is required")


def bearer_headers(config: dict, product_name: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {resolve_pat(config, product_name)}",
        "Accept": "application/json",
    }


def tls_verify(config: dict) -> bool | str:
    validate_tls_ca_bundle(config)
    ca_bundle = str(config.get("tls_ca_bundle") or "").strip()
    return ca_bundle or True


def validate_tls_ca_bundle(config: dict) -> None:
    ca_bundle = str(config.get("tls_ca_bundle") or "").strip()
    if not ca_bundle:
        return
    path = Path(ca_bundle).expanduser()
    if not path.is_file() or not os.access(path, os.R_OK):
        raise ValueError(f"TLS CA bundle does not exist or is not readable: {ca_bundle}")


def require_https_base_url(base_url: str, product_name: str) -> None:
    if urlsplit(base_url).scheme != "https":
        raise ValueError(f"{product_name} base_url must use HTTPS when storing authenticated source credentials")


async def get_with_rate_limit_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    product_name: str,
    params: dict | None = None,
    limiter: AtlassianRequestLimiter | None = None,
    zero_quota_message: str | None = None,
) -> httpx.Response:
    """GET an Atlassian REST URL with bounded HTTP 429 retry handling."""
    return await request_with_rate_limit_retry(
        client,
        "GET",
        url,
        product_name=product_name,
        params=params,
        limiter=limiter,
        zero_quota_message=zero_quota_message,
    )


async def request_with_rate_limit_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    product_name: str,
    params: dict | None = None,
    json_body: dict | None = None,
    limiter: AtlassianRequestLimiter | None = None,
    zero_quota_message: str | None = None,
) -> httpx.Response:
    """Request an Atlassian REST URL with bounded HTTP 429 retry handling."""
    for attempt in range(1, ATLASSIAN_REQUEST_ATTEMPTS + 1):
        kwargs = {}
        if params is not None:
            kwargs["params"] = params
        if json_body is not None:
            kwargs["json"] = json_body
        try:
            if limiter is not None:
                resp = await limiter.run(
                    lambda: client.request(method, url, **kwargs),
                    delay_for_response=lambda response: _limiter_delay_seconds(response, attempt),
                )
            else:
                resp = await client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            if attempt == ATLASSIAN_REQUEST_ATTEMPTS:
                detail = str(exc) or type(exc).__name__
                raise AtlassianRequestTransportError(
                    f"{product_name} request failed to reach {url} after {ATLASSIAN_REQUEST_ATTEMPTS} attempts "
                    f"({detail}). The instance may be slow or unreachable; check VPN or network, then retry."
                ) from exc
            logger.warning(
                "%s request transport error for %s (%s); retrying in %.1fs",
                product_name,
                url,
                type(exc).__name__,
                ATLASSIAN_TRANSPORT_RETRY_DELAY_SECONDS,
            )
            await asyncio.sleep(ATLASSIAN_TRANSPORT_RETRY_DELAY_SECONDS)
            continue
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp

        if _is_zero_quota_rate_limit(resp):
            raise AtlassianZeroQuotaRateLimitError(
                zero_quota_message
                or f"{product_name} REST API quota is zero for this request. "
                f"Ask the {product_name} administrator to enable REST API quota."
            )

        if attempt == ATLASSIAN_REQUEST_ATTEMPTS:
            raise AtlassianRateLimitError(
                f"{product_name} rate limit persisted after "
                f"{ATLASSIAN_REQUEST_ATTEMPTS} attempts for {url}. "
                "Retry later or reduce the source sync frequency."
            )

        delay = _retry_delay_seconds(resp, attempt)
        logger.warning(
            "%s rate limited %s; retrying in %.1f seconds",
            product_name,
            url,
            delay,
        )
        await asyncio.sleep(delay)

    raise RuntimeError("Atlassian request retry loop exited unexpectedly")


@asynccontextmanager
async def stream_with_rate_limit_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    product_name: str,
    limiter: AtlassianRequestLimiter | None = None,
    zero_quota_message: str | None = None,
):
    """Open one streaming Atlassian response with bounded retry ownership."""

    for attempt in range(1, ATLASSIAN_REQUEST_ATTEMPTS + 1):
        response: httpx.Response | None = None
        try:
            request = client.build_request(method, url)

            async def operation() -> httpx.Response:
                return await client.send(request, stream=True)

            if limiter is not None:
                response = await limiter.run(
                    operation,
                    delay_for_response=lambda item: _limiter_delay_seconds(item, attempt),
                )
            else:
                response = await operation()
        except httpx.TransportError as exc:
            if attempt == ATLASSIAN_REQUEST_ATTEMPTS:
                detail = str(exc) or type(exc).__name__
                raise AtlassianRequestTransportError(
                    f"{product_name} request failed to reach {url} after "
                    f"{ATLASSIAN_REQUEST_ATTEMPTS} attempts ({detail}). "
                    "The instance may be slow or unreachable; check VPN or network, then retry."
                ) from exc
            await asyncio.sleep(ATLASSIAN_TRANSPORT_RETRY_DELAY_SECONDS)
            continue

        try:
            if response.status_code != 429:
                response.raise_for_status()
                yield response
                return
            if _is_zero_quota_rate_limit(response):
                raise AtlassianZeroQuotaRateLimitError(
                    zero_quota_message or f"{product_name} REST API quota is zero for this request."
                )
            if attempt == ATLASSIAN_REQUEST_ATTEMPTS:
                raise AtlassianRateLimitError(
                    f"{product_name} rate limit persisted after {ATLASSIAN_REQUEST_ATTEMPTS} attempts for {url}."
                )
            await asyncio.sleep(_retry_delay_seconds(response, attempt))
        finally:
            await response.aclose()

    raise RuntimeError("Atlassian streaming request retry loop exited unexpectedly")


def _limiter_delay_seconds(resp: httpx.Response, attempt: int) -> float | None:
    if resp.status_code == 429:
        return _retry_delay_seconds(resp, attempt)
    retry_after = str(getattr(resp, "headers", {}).get("Retry-After") or "").strip()
    if 500 <= resp.status_code <= 599 and retry_after:
        return _retry_delay_seconds(resp, attempt)
    return None


def _is_zero_quota_rate_limit(resp: httpx.Response) -> bool:
    headers = getattr(resp, "headers", {})
    limit = str(headers.get("X-RateLimit-Limit") or headers.get("x-ratelimit-limit") or "").strip()
    fillrate = str(headers.get("X-RateLimit-FillRate") or headers.get("x-ratelimit-fillrate") or "").strip()
    return resp.status_code == 429 and limit == "0" and fillrate == "0"


def _retry_delay_seconds(resp: httpx.Response, attempt: int) -> float:
    retry_after = str(getattr(resp, "headers", {}).get("Retry-After") or "").strip()
    if retry_after:
        if retry_after.isdigit():
            return min(float(retry_after), ATLASSIAN_MAX_RETRY_DELAY_SECONDS)
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
            return min(max(seconds, 0.0), ATLASSIAN_MAX_RETRY_DELAY_SECONDS)
        except (TypeError, ValueError):
            pass
    return min(
        ATLASSIAN_DEFAULT_RETRY_DELAY_SECONDS * attempt,
        ATLASSIAN_MAX_RETRY_DELAY_SECONDS,
    )
