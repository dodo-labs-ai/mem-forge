"""In-memory model endpoints behind the HTTP clients LiteLLM sends requests through."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from memforge.llm.transport import NoRedirectAsyncClient

Respond = Callable[[httpx.Request], httpx.Response]


def send_every_new_client_to(monkeypatch, respond: Respond) -> None:
    """Have every HTTP client built from now on send its requests to ``respond``.

    LiteLLM and the provider SDKs it drives send every request through an
    ``httpx.AsyncClient``, whichever transport they would give it. Each new
    client gets ``respond`` as its transport instead, so no request leaves the
    process, and the client itself still follows or refuses redirects. A client
    built with an in-memory transport of its own keeps it.
    """
    build = httpx.AsyncClient.__init__

    def __init__(self: httpx.AsyncClient, *args, **kwargs) -> None:
        if not isinstance(kwargs.get("transport"), httpx.MockTransport):
            kwargs.update(transport=httpx.MockTransport(respond), mounts=None, trust_env=False)
        build(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", __init__)


class InMemorySharedClient(NoRedirectAsyncClient):
    """The shared client with each connection pool replaced by an in-memory server."""

    def __init__(self, respond: Respond) -> None:
        super().__init__()
        self.respond = respond
        self.requests: list[httpx.Request] = []
        self.pools_opened = 0

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def _new_pool(self) -> httpx.AsyncClient:
        self.pools_opened += 1
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle), follow_redirects=False)


class OllamaLookups:
    """Stand in for the client LiteLLM asks an Ollama server about a model with.

    It records each question and answers none, as when no server is listening.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str]]] = []

    def post(self, url: str, json=None, headers=None, **_options) -> httpx.Response:
        self.requests.append((url, dict(headers or {})))
        raise httpx.ConnectError("connection refused")
