"""Storage-neutral contract for admin source endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol, cast, runtime_checkable

SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES = 1440
SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES = 5
SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES = 10080
SourceListSortMode = Literal["newest", "name", "recently_synced"]
SOURCE_LIST_DEFAULT_SORT_MODE: SourceListSortMode = "newest"
SOURCE_LIST_SORT_MODES = frozenset({"newest", "name", "recently_synced"})


def validate_source_list_sort_mode(value: str) -> SourceListSortMode:
    if value not in SOURCE_LIST_SORT_MODES:
        raise ValueError(f"Unsupported source-list sort mode: {value}")
    return cast(SourceListSortMode, value)


@runtime_checkable
class SourceAdminReader(Protocol):
    async def list_sources(self) -> list[dict[str, Any]]: ...

    async def list_searchable_source_ids_for_user(
        self,
        source_ids: list[str],
        user_id: str,
    ) -> set[str]: ...

    async def count_source_memories(
        self,
        source_id: str,
        *,
        include_private: bool = False,
        owner_user_id: str | None = None,
    ) -> int:
        """Count search-visible active memories linked to ``source_id``.

        This is a source total. Per-user source subscription state is returned
        separately by source-list APIs.
        """
        ...

    async def count_documents(self, source: str | None = None) -> int: ...

    async def get_sync_history(
        self, source: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]: ...

    async def get_latest_source_sync_run(
        self,
        *,
        source_id: str,
        workspace_id: str = "default",
    ) -> Any | None: ...

    async def get_source(self, source_id: str) -> dict[str, Any] | None: ...

    async def list_source_projects(
        self,
        source_id: str,
        *,
        include_private: bool = False,
        owner_user_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def is_source_enabled_for_user(self, source_id: str, user_id: str) -> bool: ...

    async def set_source_subscription(
        self, source_id: str, user_id: str, enabled: bool
    ) -> None: ...

    async def is_source_pinned_for_user(self, source_id: str, user_id: str) -> bool: ...

    async def set_source_pinned_for_user(
        self, source_id: str, user_id: str, pinned: bool
    ) -> None: ...

    async def get_source_list_sort_mode(self, user_id: str) -> SourceListSortMode: ...

    async def set_source_list_sort_mode(
        self, user_id: str, sort_mode: SourceListSortMode
    ) -> None: ...

    async def set_source_sync_schedule(
        self,
        source_id: str,
        *,
        enabled: bool,
        interval_minutes: int,
        next_run_at: datetime | None = None,
    ) -> None: ...

    async def claim_due_scheduled_sources(
        self,
        *,
        now: datetime | None = None,
        limit: int = 50,
        exclude_source_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]: ...
