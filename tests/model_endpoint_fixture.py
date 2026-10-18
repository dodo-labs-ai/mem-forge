"""In-memory model endpoints behind the HTTP clients LiteLLM sends requests through."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from memforge.llm.transport import NoRedirectAsyncClient

Respond = Callable[[httpx.Request], httpx.Response]


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
