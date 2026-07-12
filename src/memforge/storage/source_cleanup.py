"""Durable cleanup of document artifacts owned by deleted sources."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from memforge.models import SourceArtifactCleanupTask
from memforge.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)


class SourceArtifactCleanupStore(Protocol):
    async def list_source_artifact_cleanup_tasks(
        self,
        *,
        limit: int = 100,
    ) -> list[SourceArtifactCleanupTask]: ...

    async def complete_source_artifact_cleanup_task(self, task_id: str) -> None: ...

    async def fail_source_artifact_cleanup_task(self, task_id: str, error: str) -> None: ...


class SourceArtifactCleanupService:
    """Process the source-deletion outbox against an exact-URI document store."""

    def __init__(
        self,
        store: SourceArtifactCleanupStore,
        document_store: DocumentStore,
    ) -> None:
        self._store = store
        self._document_store = document_store

    async def run_pending(self, *, limit: int = 100) -> int:
        completed = 0
        for task in await self._store.list_source_artifact_cleanup_tasks(limit=limit):
            try:
                await asyncio.to_thread(
                    self._document_store.delete_artifact,
                    task.artifact_uri,
                )
            except Exception as exc:
                logger.warning("Artifact cleanup failed for task %s: %s", task.task_id, exc)
                await self._store.fail_source_artifact_cleanup_task(task.task_id, str(exc))
                continue
            await self._store.complete_source_artifact_cleanup_task(task.task_id)
            completed += 1
        return completed
