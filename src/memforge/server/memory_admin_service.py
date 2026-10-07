"""Canonical admin memory list behavior.

The HTTP route owns request parsing and response shaping; storage adapters own
query execution. This service keeps the shared route behavior independent from
SQLite, HANA, or future database-specific SQL.
"""

from __future__ import annotations

from typing import Protocol

from memforge.memory.cross_document_relation_reader import RelationReadStore, read_memory_relations
from memforge.storage.adapters.context import AccessScope
from memforge.server.review_admin_service import read_open_review_ids
from memforge.storage.admin_review import ReviewAdminReader
from memforge.storage.admin_memory import (
    MemoryAdminListFilters,
    MemoryAdminPage,
    MemoryAdminPageReader,
)


class MemoryAdminListStore(MemoryAdminPageReader, RelationReadStore, ReviewAdminReader, Protocol):
    """Reads one admin list page with each Memory's origin, Sources, relations and open Review."""


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
    reader: MemoryAdminListStore,
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
    memory_ids = [memory.id for memory in page.memories]
    pairs = await reader.get_origin_source_pairs(memory_ids)
    return MemoryAdminPage(
        memories=page.memories,
        total=page.total,
        origins={
            memory_id: pick_origin_source_type(memory_pairs)
            for memory_id, memory_pairs in pairs.items()
        },
        sources=await reader.get_memory_source_refs_many(memory_ids, scope),
        relations=await read_memory_relations(reader, memory_ids, scope),
        open_reviews=await read_open_review_ids(reader, page.memories, scope=scope),
    )


async def count_project_memories(
    reader: MemoryAdminPageReader,
    *,
    scope: AccessScope,
    project_keys: list[str],
) -> dict[str, int]:
    """Count the memories in each project that the caller can see.

    Each count is the total the memory list reports when filtered by that
    project under the same scope, so the two never disagree.
    """
    counts = await reader.count_memory_admin_projects(scope=scope)
    return {project_key: counts.get(project_key, 0) for project_key in project_keys}
