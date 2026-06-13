"""Storage-neutral contract for the admin memory list endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from memforge.models import Memory
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

    async def get_origin_source_pairs(
        self,
        memory_ids: list[str],
    ) -> dict[str, list[tuple[str, str | None, str | None]]]: ...
