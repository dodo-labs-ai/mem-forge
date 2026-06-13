"""Canonical admin memory list behavior.

The HTTP route owns request parsing and response shaping; storage adapters own
query execution. This service keeps the shared route behavior independent from
SQLite, HANA, or future database-specific SQL.
"""

from __future__ import annotations

from memforge.storage.adapters.context import AccessScope
from memforge.storage.admin_memory import (
    MemoryAdminListFilters,
    MemoryAdminPage,
    MemoryAdminPageReader,
)


def pick_origin_source_type(
    pairs: list[tuple[str, str | None, str | None]],
) -> tuple[str | None, str | None]:
    if not pairs:
        return None, None
    for source_type, support_kind, client in pairs:
        if support_kind == "extracted":
            return source_type, client
    return pairs[0][0], pairs[0][2]


async def list_memory_admin_page(
    reader: MemoryAdminPageReader,
    *,
    scope: AccessScope,
    filters: MemoryAdminListFilters,
    limit: int,
    offset: int,
) -> MemoryAdminPage:
    page = await reader.query_memory_admin_page(
        scope=scope,
        filters=filters,
        limit=limit,
        offset=offset,
    )
    pairs = await reader.get_origin_source_pairs([memory.id for memory in page.memories])
    return MemoryAdminPage(
        memories=page.memories,
        total=page.total,
        origins={
            memory_id: pick_origin_source_type(memory_pairs)
            for memory_id, memory_pairs in pairs.items()
        },
    )
