"""Transport policy shared by every model call made through the LiteLLM SDK.

Generation and embedding both reach their providers through the in-process
LiteLLM SDK. This module holds what the two paths share: HTTP clients that
refuse redirects and end each exchange at its timeout, the SDK's own retries
and help notice switched off, and the rule for which provider failures are
worth another attempt. Each caller owns its attempt loop, its pauses and its
logical deadline.

LiteLLM sends a request through one of two kinds of HTTP client. Its
OpenAI-compatible routes, which carry every model of an OpenAI-compatible
endpoint, share one process-wide client. Every other route, ``anthropic/``
and ``openrouter/`` among them, uses a client LiteLLM builds for that
provider. Both kinds refuse redirects here, and both end a request at the
timeout it was sent with.

That timeout is a deadline for the whole exchange. An HTTP client applies a
timeout to each wait on its own: connecting, writing, every read. A provider
that keeps the connection busy, sending a byte now and then while the model
is still working, never makes one wait that long, so the request would stay
open for as long as the provider likes. Here the exchange as a whole fails as
a read timeout when its timeout runs out, which the SDK reports as a timeout
and ``is_retryable_provider_error`` counts as transient. Some routes do not
send the caller's timeout at all, so every caller also bounds its own call
with its logical deadline.

LiteLLM also looks a model up by itself, to learn its limits and prices. For an
Ollama model that lookup is a request, with the key the process environment
holds for Ollama. It is allowed to go to the server the environment names and
nowhere else.

Importing this module applies the process-wide policy. Every module that issues
a LiteLLM request imports it, so the policy is in force before LiteLLM builds a
provider client in any process, server or command line. MemForge issues every
model request asynchronously, so the policy covers the asynchronous clients.
"""

from __future__ import annotations

import asyncio
import functools
import math
import threading
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import ModelInfoBase

__all__ = [
    "NoRedirectAsyncClient",
    "disable_sdk_default_retries",
    "end_provider_client_exchanges_at_their_timeout",
    "install_no_redirect_http_client",
    "is_connection_failure",
    "is_retryable_provider_error",
    "keep_ollama_model_lookups_off_request_endpoints",
    "provider_retry_after_seconds",
    "provider_status_code",
    "refuse_redirects_in_provider_clients",
    "send_within_timeout",
    "silence_sdk_help_notice",
]

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
_REFUSES_REDIRECTS_MARK = "_memforge_refuses_redirects"
_ENDS_EXCHANGES_AT_THEIR_TIMEOUT_MARK = "_memforge_ends_exchanges_at_their_timeout"
_LOOKS_UP_AT_ENVIRONMENT_SERVER_ONLY_MARK = "_memforge_looks_up_at_environment_server_only"


_Send = Callable[..., Awaitable[httpx.Response]]


async def send_within_timeout(send: _Send, request: httpx.Request, **kwargs: Any) -> httpx.Response:
    """Send ``request`` with ``send`` and have the answer within the request's timeout.

    The timeout counts from sending the request to the last byte of an answer
    that is read whole. A streamed answer is bounded until its headers arrive,
    and whoever reads its body bounds the rest. A request sent with no timeout
    is sent unbounded.
    """
    seconds = _exchange_timeout_s(request)
    if seconds is None:
        return await send(request, **kwargs)
    bound = asyncio.timeout(seconds)
    try:
        async with bound:
            return await send(request, **kwargs)
    except TimeoutError as exc:
        if not bound.expired():
            raise
        raise httpx.ReadTimeout(
            f"No complete answer within the request timeout of {seconds:g}s",
            request=request,
        ) from exc


def _exchange_timeout_s(request: httpx.Request) -> float | None:
    """The longest wait the request allows, which is how long its whole exchange may take."""
    waits = (request.extensions.get("timeout") or {}).values()
    limits = [float(wait) for wait in waits if wait is not None]
    return max(limits) if limits else None


class NoRedirectAsyncClient(httpx.AsyncClient):
    """An HTTP client that never follows a redirect, safe to share in a process.

    A followed redirect resends the request, with its credentials and its
    payload, to a location the operator never configured. A model endpoint that
    answers with a redirect is therefore a failed request, not a new address.

    LiteLLM hands this one object to every OpenAI-compatible provider client
    it builds, on whichever event loop is running, while a pooled connection
    only works on the loop that opened it. Each event loop therefore sends
    through its own connection pool.

    Each exchange ends at the timeout the request was sent with.
    """

    def __init__(self) -> None:
        super().__init__(follow_redirects=False)
        self._pools: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}
        self._pools_lock = threading.Lock()

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        kwargs["follow_redirects"] = False
        return await send_within_timeout(self._pool().send, request, **kwargs)

    async def aclose(self) -> None:
        with self._pools_lock:
            pool = self._pools.pop(asyncio.get_running_loop(), None)
        if pool is not None:
            await pool.aclose()
        await super().aclose()

    def _pool(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        with self._pools_lock:
            pool = self._pools.get(loop)
            if pool is None:
                for known in [known for known in self._pools if known.is_closed()]:
                    del self._pools[known]
                pool = self._pools[loop] = self._new_pool()
            return pool

    def _new_pool(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(follow_redirects=False)


def install_no_redirect_http_client() -> httpx.AsyncClient:
    """Make LiteLLM's shared HTTP client one that refuses redirects.

    A client that already refuses redirects is kept, so a host process may
    install its own before importing MemForge; that client's timeouts are the
    host's to choose.
    """
    session = litellm.aclient_session
    if isinstance(session, httpx.AsyncClient) and not session.follow_redirects:
        return session
    session = NoRedirectAsyncClient()
    litellm.aclient_session = session
    return session


def refuse_redirects_in_provider_clients() -> None:
    """Make every HTTP client LiteLLM builds for a provider refuse redirects.

    LiteLLM builds these clients in ``AsyncHTTPHandler.create_client`` and sets
    them to follow redirects. Wrapping that one method turns the setting off on
    each client as it is built, whichever provider it is for.
    """
    build_client = AsyncHTTPHandler.create_client
    if getattr(build_client, _REFUSES_REDIRECTS_MARK, False):
        return

    @functools.wraps(build_client)
    def create_client(self: AsyncHTTPHandler, *args: Any, **kwargs: Any) -> httpx.AsyncClient:
        client = build_client(self, *args, **kwargs)
        client.follow_redirects = False
        return client

    setattr(create_client, _REFUSES_REDIRECTS_MARK, True)
    AsyncHTTPHandler.create_client = create_client


def end_provider_client_exchanges_at_their_timeout() -> None:
    """Make every HTTP client LiteLLM builds for a provider end a request at its timeout.

    LiteLLM sends each request of a provider route through ``send`` of the
    client ``AsyncHTTPHandler.create_client`` builds. Each client gets that
    method bounded by ``send_within_timeout`` as it is built.
    """
    build_client = AsyncHTTPHandler.create_client
    if getattr(build_client, _ENDS_EXCHANGES_AT_THEIR_TIMEOUT_MARK, False):
        return

    @functools.wraps(build_client)
    def create_client(self: AsyncHTTPHandler, *args: Any, **kwargs: Any) -> httpx.AsyncClient:
        client = build_client(self, *args, **kwargs)
        client.send = functools.partial(send_within_timeout, client.send)
        return client

    setattr(create_client, _ENDS_EXCHANGES_AT_THEIR_TIMEOUT_MARK, True)
    AsyncHTTPHandler.create_client = create_client


def keep_ollama_model_lookups_off_request_endpoints() -> None:
    """Stop LiteLLM asking the endpoint of an Ollama request about its model.

    LiteLLM asks an Ollama server for a model's limits, and sends the key
    ``OLLAMA_API_KEY`` holds with the question. Asked with no address, it asks
    the server ``OLLAMA_API_BASE`` names: the environment paired that server
    with that key. After a request it asks the address the request went to
    instead, an address the operator configured with a key of its own or with
    none. That question is not sent, and the lookup reports what it reports
    when a server does not answer: a model with no known limits.
    """
    look_up = litellm.OllamaConfig.get_model_info
    if getattr(look_up, _LOOKS_UP_AT_ENVIRONMENT_SERVER_ONLY_MARK, False):
        return

    @functools.wraps(look_up)
    def get_model_info(self: Any, model: str, api_base: str | None = None) -> ModelInfoBase:
        if api_base is None:
            return look_up(self, model)
        return ModelInfoBase(
            key=model,
            litellm_provider="ollama",
            mode="chat",
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
            max_tokens=None,
            max_input_tokens=None,
            max_output_tokens=None,
        )

    setattr(get_model_info, _LOOKS_UP_AT_ENVIRONMENT_SERVER_ONLY_MARK, True)
    litellm.OllamaConfig.get_model_info = get_model_info


def disable_sdk_default_retries() -> None:
    """Make a request that asks the SDK for no retries get none.

    Callers pass a retry count of zero with every request. LiteLLM's OpenAI
    embedding path reads zero as "not set" and falls back to the SDK-wide
    default, so that default is zero as well.
    """
    litellm.DEFAULT_MAX_RETRIES = 0


def silence_sdk_help_notice() -> None:
    """Stop LiteLLM printing its help notice to standard output on a failed request.

    A failed request is reported through the caller's own error and log line.
    """
    litellm.suppress_debug_info = True


def provider_status_code(exc: BaseException) -> int | None:
    """Return the HTTP status the provider answered with, when it answered.

    On some routes LiteLLM reports any provider answer as a connection error
    with a status of its own. The answer's status is then the one on the HTTP
    error underneath, so a rejected request is not mistaken for a server fault.
    A provider client that raises httpx's own status error carries the status
    on its response.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    if isinstance(exc, litellm.APIConnectionError):
        answered = _answered_status_code(exc)
        if answered is not None:
            return answered
    raw = getattr(exc, "status_code", None)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def provider_retry_after_seconds(exc: BaseException) -> float | None:
    """Return how long the provider asked to wait before another attempt.

    Reads a ``Retry-After`` header given in seconds from the failed response.
    """
    headers = getattr(exc, "litellm_response_headers", None)
    if headers is None:
        return None
    try:
        seconds = float(headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def is_retryable_provider_error(exc: BaseException) -> bool:
    """Whether a failed provider request may succeed when sent again unchanged.

    Timeouts, connection failures, request timeout (408), conflict (409), rate
    limiting (429) and server errors (5xx) are transient, whether LiteLLM or
    an httpx client reports them. Every other failure repeats for the same
    request.
    """
    if isinstance(
        exc,
        (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError),
    ):
        return True
    status_code = provider_status_code(exc)
    return status_code in _RETRYABLE_STATUS_CODES or bool(
        status_code is not None and status_code >= 500
    )


def is_connection_failure(exc: BaseException) -> bool:
    """Whether the request never reached the provider, anywhere in the error's chain."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _answered_status_code(exc: BaseException) -> int | None:
    seen: set[int] = set()
    current: BaseException | None = exc.__cause__ or exc.__context__
    while current is not None and id(current) not in seen:
        if isinstance(current, httpx.HTTPStatusError):
            return current.response.status_code
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


install_no_redirect_http_client()
refuse_redirects_in_provider_clients()
end_provider_client_exchanges_at_their_timeout()
keep_ollama_model_lookups_off_request_endpoints()
disable_sdk_default_retries()
silence_sdk_help_notice()
