"""Storage-neutral contract for the admin memory list endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from memforge.models import Memory, MemoryRelationContext, MemorySourceRef
from memforge.storage.adapters.context import AccessScope


@dataclass(frozen=True)
class MemoryAdminListFilters:
    memory_type: str | None = None
    status: str | None = None
    source: str | None = None
    project: str | None = None
    search: str | None = None


@dataclass(frozen=True)
class MemoryAdminQueryPage:
    memories: list[Memory]
    total: int


@dataclass(frozen=True)
class MemoryAdminPage:
    memories: list[Memory]
    total: int
    origins: dict[str, tuple[str | None, str | None]]
    sources: Mapping[str, tuple[MemorySourceRef, ...]]
    relations: Mapping[str, tuple[MemoryRelationContext, ...]]


@runtime_checkable
class MemoryAdminPageReader(Protocol):
    async def query_memory_admin_page(
        self,
        *,
        scope: AccessScope,
        filters: MemoryAdminListFilters,
        limit: int,
        offset: int,
    ) -> MemoryAdminQueryPage: ...

    async def count_memory_admin_projects(self, *, scope: AccessScope) -> Mapping[str, int]:
        """Count the Memories under ``scope`` per project key, in one grouped read.

        Each count equals the ``total`` of ``query_memory_admin_page`` filtered
        by that project with no other filter, so the two never disagree. Keys
        without Memories may be left out.
        """
        ...

    async def get_origin_source_pairs(
        self,
        memory_ids: list[str],
    ) -> dict[str, list[tuple[str, str | None, str | None]]]: ...
