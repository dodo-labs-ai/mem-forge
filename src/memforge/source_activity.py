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
    """Raised when a Source-scoped mutation cannot acquire admission."""


@dataclass(frozen=True)
class SourceActivityLease:
    id: str
    source_id: str
    kind: SourceActivityKind
    epoch: int
    capability: str | None
    lease_until: datetime
