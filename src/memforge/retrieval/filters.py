"""Structured search filters shared by API, retrieval, and storage adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


MemoryTimeRangeType = Literal["source_updated_at", "memory_updated_at"]


@dataclass(frozen=True)
class MemorySourceFilter:
    """Exact source facets for memory search.

    Empty tuples mean "no narrowing" for that facet. Values are validated at
    the request boundary; storage implementations only apply exact matches.
    """

    source_types: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    clients: tuple[str, ...] = ()
    repo_identifiers: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not (
            self.source_types
            or self.sources
            or self.clients
            or self.repo_identifiers
        )


@dataclass(frozen=True)
class MemoryTimeRange:
    """Exact date-window filter for memory search.

    ``after`` and ``before`` are internal UTC instants converted from the
    public date-only request shape. ``source_updated_at`` means the source item
    linked to the memory was observed/updated in the requested window; unbacked
    memories without a provenance row do not match that mode.
    ``memory_updated_at`` means the MemForge memory row itself changed in the
    requested window.
    """

    after: datetime | None = None
    before: datetime | None = None
    date_type: MemoryTimeRangeType = "source_updated_at"

    def is_empty(self) -> bool:
        return self.after is None and self.before is None
