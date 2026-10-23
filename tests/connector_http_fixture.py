"""A fake provider behind a connector's ``ConnectorHttpClient``: one transport and one sleep that returns at once.

A Gene builds its client in ``authenticate``. :meth:`FakeConnectorHttp.serving`
hands that client the fake's ``httpx.MockTransport`` and stub sleep, so the
real Gene and the real retry loop run against the fake without waiting::

    http = FakeConnectorHttp(handler)
    with http.serving(jira_gene):
        await gene.authenticate()
    ...
    assert http.waits == [30.0]           # what the client and its limiter waited for
    assert all(client.is_closed for client in http.clients)

A test that sets a Gene's client by hand builds it with :meth:`client`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any
from unittest.mock import patch

import httpx

from memforge.genes.toolkit.http import ConnectorHttpClient

Handler = Callable[[httpx.Request], Any]


class FakeConnectorHttp:
    """Answers every ``ConnectorHttpClient`` built through it with ``handler``, and records each wait."""

    def __init__(self, handler: Handler) -> None:
        self.transport = httpx.MockTransport(handler)
        self.waits: list[float] = []
        self.clients: list[ConnectorHttpClient] = []

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)

    def build(self, **options: Any) -> ConnectorHttpClient:
        """A client with the Gene's own ``options``, answered by the fake."""
        client = ConnectorHttpClient(**options, transport=self.transport, sleep=self.sleep)
        self.clients.append(client)
        return client

    def client(self, base_url: str, **options: Any) -> ConnectorHttpClient:
        """A client for a test to set on a Gene by hand; ``options`` override the defaults."""
        return self.build(
            **{"product_name": "Test", "timeout": 5.0, "follow_redirects": True, "base_url": base_url, **options}
        )

    @contextmanager
    def serving(self, gene_module: ModuleType) -> Iterator[FakeConnectorHttp]:
        """While the block runs, every ``ConnectorHttpClient`` that ``gene_module`` builds is answered by the fake."""
        with patch.object(gene_module, "ConnectorHttpClient", self.build):
            yield self
