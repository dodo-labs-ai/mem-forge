"""Durable admission contract for Source-scoped mutation activity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SourceActivityKind(str, Enum):
    SYNC = "sync"
    EXTERNAL_COLLECTION = "external_collection"
    AGENT_PATCH = "agent_patch"
    MAINTENANCE = "maintenance"


class SourceActivityConflict(ValueError):
    """Raised when a Source-scoped mutation cannot acquire admission.

    Work that loses its admission may succeed once it holds the Source again,
    so the failure is retryable.
    """

    retryable = True


class SourceSyncRunActive(SourceActivityConflict):
    """Raised when a change the Source's pending or running sync run fences is requested.

    An operator reprocess, and every Source change other than a pause, waits
    for the run to end. ``run_id`` names the run, so that the caller can stop
    it and try again.
    """

    def __init__(self, message: str, *, run_id: str) -> None:
        super().__init__(message)
        self.run_id = run_id


SOURCE_SYNC_CANCELLED_BY_USER = "cancelled_by_user"
"""The error a sync run or local-agent job a person stopped fails with."""


class SourceSyncHeartbeat(str, Enum):
    """What a durable sync run's heartbeat found.

    ``RENEWED``: the worker holds the run and its lease was renewed.
    ``STOP_REQUESTED``: the worker holds the run and its lease was renewed, and a
    person asked to stop it; the worker stops and fails the run.
    ``LOST``: the worker no longer holds the run, and stops without writing.
    """

    RENEWED = "renewed"
    STOP_REQUESTED = "stop_requested"
    LOST = "lost"


@dataclass(frozen=True)
class SourceActivityLease:
    id: str
    source_id: str
    kind: SourceActivityKind
    capability: str | None
    lease_until: datetime
