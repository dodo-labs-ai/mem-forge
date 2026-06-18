"""Structured search filters shared by API, retrieval, and storage adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySourceFilter:
    """Exact source facets for memory search.

    Empty tuples mean "no narrowing" for that facet. Values are validated at
    the request boundary; storage implementations only apply exact matches.
    """

    source_types: tuple[str, ...] = ()
    clients: tuple[str, ...] = ()
    repo_identifiers: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not (
            self.source_types
            or self.clients
            or self.repo_identifiers
        )
