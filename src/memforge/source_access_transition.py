"""Durable orchestration for whole-Source access changes."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from memforge.source_access import SourceAccessPolicy, source_owner_user_id

logger = logging.getLogger(__name__)


class SourceAccessTransitionError(RuntimeError):
    """Raised when a Source access command cannot be completed."""


class SourceAccessTransitionService:
    """Keep Source access changes fail-closed across DB and vector projections."""

    def __init__(self, *, db: Any, memory_store: Any, sync_service: Any | None = None) -> None:
        self.db = db
        self.memory_store = memory_store
        self.sync_service = sync_service

    async def start(
        self,
        *,
        source_id: str,
        actor_user_id: str,
        target_policy: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key is required")
        SourceAccessPolicy(target_policy)
        transition = await self.db.create_source_access_transition(
            operation_id=f"sat-{uuid.uuid4().hex}",
            source_id=source_id,
            idempotency_key=idempotency_key.strip(),
            actor_user_id=actor_user_id,
            target_policy=target_policy,
        )
        if self.sync_service is not None:
            await self.sync_service.cancel_source(source_id)
        return transition

    async def run(self, operation_id: str) -> dict[str, Any]:
        transition = await self._require_transition(operation_id)
        if transition["status"] == "completed":
            return transition
        source = await self.db.get_source(transition["source_id"])
        if source is None:
            raise SourceAccessTransitionError("source_not_found")
        await self.db.mark_source_access_transition_running(operation_id)
        try:
            memory_ids = await self.db.reconcile_source_memory_access(
                operation_id=operation_id,
                source_id=transition["source_id"],
                target_policy=transition["target_policy"],
                source_owner_user_id=source_owner_user_id(source),
            )
            for memory_id in memory_ids:
                await self.memory_store.reindex_memory_access(memory_id)
                await self.db.advance_source_access_transition_progress(operation_id)
            await self.db.complete_source_access_transition(operation_id)
        except Exception as exc:
            logger.exception("Source access transition failed: %s", operation_id)
            await self.db.mark_source_access_transition_failed(
                operation_id,
                error_code="source_access_reconciliation_failed",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            raise
        completed = await self._require_transition(operation_id)
        return completed

    async def retry(self, operation_id: str) -> dict[str, Any]:
        transition = await self._require_transition(operation_id)
        if transition["status"] != "failed":
            raise SourceAccessTransitionError("source_access_transition_not_retryable")
        return await self.run(operation_id)

    async def revert(self, operation_id: str) -> dict[str, Any]:
        transition = await self._require_transition(operation_id)
        if transition["status"] != "failed":
            raise SourceAccessTransitionError("source_access_transition_not_revertible")
        source = await self.db.get_source(transition["source_id"])
        if source is None:
            raise SourceAccessTransitionError("source_not_found")
        await self.db.mark_source_access_transition_running(operation_id)
        try:
            memory_ids = await self.db.reconcile_source_memory_access(
                operation_id=operation_id,
                source_id=transition["source_id"],
                target_policy=transition["previous_policy"],
                source_owner_user_id=source_owner_user_id(source),
            )
            for memory_id in memory_ids:
                await self.memory_store.reindex_memory_access(memory_id)
                await self.db.advance_source_access_transition_progress(operation_id)
            await self.db.mark_source_access_transition_reverted(operation_id)
        except Exception as exc:
            logger.exception("Source access transition revert failed: %s", operation_id)
            await self.db.mark_source_access_transition_failed(
                operation_id,
                error_code="source_access_revert_failed",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            raise
        return await self._require_transition(operation_id)

    async def _require_transition(self, operation_id: str) -> dict[str, Any]:
        transition = await self.db.get_source_access_transition(operation_id)
        if transition is None:
            raise SourceAccessTransitionError("source_access_transition_not_found")
        return transition
