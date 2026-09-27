"""Durable cleanup of stored document objects that a Source released.

A task names one exact object URI. Releasing an object only queues it: object
keys are written in place per Source and Document, so the same key can be
written and named again (a Document renamed back, a removed Document that
returns) before cleanup runs. Cleanup therefore decides at run time. It holds
the Source's activity lease, which every sync of that Source holds while it
writes objects and records their references, checks that no stored input or
retained sync input names the URI, and only then deletes the object.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from typing import Any, Protocol

from memforge.models import SourceArtifactCleanupTask
from memforge.source_activity import SourceActivityConflict, SourceActivityKind, SourceActivityLease
from memforge.storage.document_store import ArtifactNotOwnedError, DocumentStore

logger = logging.getLogger(__name__)

# Long enough for one batch of object deletions; the lease is released as soon
# as the batch finishes.
CLEANUP_LEASE_SECONDS = 300


class SourceArtifactCleanupStore(Protocol):
    async def list_source_artifact_cleanup_tasks(
        self,
        *,
        limit: int = 100,
    ) -> list[SourceArtifactCleanupTask]: ...

    async def complete_source_artifact_cleanup_task(self, task_id: str) -> None: ...

    async def fail_source_artifact_cleanup_task(self, task_id: str, error: str) -> None: ...

    async def source_artifact_uri_is_referenced(self, artifact_uri: str) -> bool: ...

    async def get_source(self, source_id: str) -> dict[str, Any] | None: ...

    async def acquire_source_activity(
        self,
        *,
        activity_id: str,
        source_id: str,
        kind: SourceActivityKind,
        capability: str | None = None,
        lease_seconds: int = 900,
        expected_epoch: int | None = None,
    ) -> SourceActivityLease: ...

    async def release_source_activity(
        self,
        *,
        activity_id: str,
        capability: str | None = None,
    ) -> bool: ...


class SourceArtifactCleanupService:
    """Process the artifact cleanup outbox against an exact-URI document store."""

    def __init__(
        self,
        store: SourceArtifactCleanupStore,
        document_store: DocumentStore,
    ) -> None:
        self._store = store
        self._document_store = document_store

    async def run_pending(self, *, limit: int = 100) -> int:
        """Process pending tasks; a Source with an active sync keeps its tasks for a later run."""

        tasks_by_source: dict[str, list[SourceArtifactCleanupTask]] = {}
        for task in await self._store.list_source_artifact_cleanup_tasks(limit=limit):
            tasks_by_source.setdefault(task.source_id, []).append(task)
        completed = 0
        for source_id, tasks in tasks_by_source.items():
            activity_id = f"source-artifact-cleanup-{uuid.uuid4().hex}"
            try:
                await self._store.acquire_source_activity(
                    activity_id=activity_id,
                    source_id=source_id,
                    kind=SourceActivityKind.MAINTENANCE,
                    capability=activity_id,
                    lease_seconds=CLEANUP_LEASE_SECONDS,
                )
            except SourceActivityConflict as exc:
                if await self._store.get_source(source_id) is not None:
                    logger.info("Artifact cleanup for %s waits for its active Source activity: %s", source_id, exc)
                    continue
                # Without a Source row nothing writes this Source's object keys.
                completed += await self._run_tasks(tasks)
                continue
            try:
                completed += await self._run_tasks(tasks)
            finally:
                await self._store.release_source_activity(activity_id=activity_id, capability=activity_id)
        return completed

    async def _run_tasks(self, tasks: Sequence[SourceArtifactCleanupTask]) -> int:
        completed = 0
        for task in tasks:
            if await self._store.source_artifact_uri_is_referenced(task.artifact_uri):
                # Named again since it was released; a later release queues it again.
                await self._store.complete_source_artifact_cleanup_task(task.task_id)
                completed += 1
                continue
            try:
                await asyncio.to_thread(
                    self._document_store.delete_artifact,
                    task.artifact_uri,
                )
            except ArtifactNotOwnedError as exc:
                logger.warning(
                    "Artifact cleanup task %s is outside this store's ownership boundary; "
                    "completing without deletion: %s",
                    task.task_id,
                    exc,
                )
                await self._store.complete_source_artifact_cleanup_task(task.task_id)
                completed += 1
                continue
            except Exception as exc:
                logger.warning("Artifact cleanup failed for task %s: %s", task.task_id, exc)
                await self._store.fail_source_artifact_cleanup_task(task.task_id, str(exc))
                continue
            await self._store.complete_source_artifact_cleanup_task(task.task_id)
            completed += 1
        return completed
