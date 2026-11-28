"""A database's own local-agent job store, with the lease each submission is checked against set by the test."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from memforge.local_agent.source_contract import LocalAgentLease
from memforge.storage.database import Database


class LeasedJobStore:
    """``db`` as its job store, except that ``current_local_agent_lease`` answers with ``lease``.

    Tests that submit outside a daemon's lease use it in place of leasing a
    job first; every other job-store call reaches the database itself.
    """

    def __init__(self, db: Database, lease: Callable[..., Awaitable[LocalAgentLease | None]]) -> None:
        self._db = db
        self._lease = lease

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)

    async def current_local_agent_lease(self, **lease: Any) -> LocalAgentLease | None:
        return await self._lease(**lease)
