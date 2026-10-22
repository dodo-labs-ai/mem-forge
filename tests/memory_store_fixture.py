"""The memory store a test sync orchestrator writes through."""

from __future__ import annotations

from typing import Any

from memforge.memory.audit import AuditContext
from memforge.memory.lifecycle_plan import LifecycleVectorDeliveryResult, LifecycleVectorDeliveryState
from memforge.source_activity import SourceActivityLease
from memforge.storage.database import Database


class OrchestratorMemoryStore:
    """The memory-store operations the sync orchestrator calls, against the relational store only.

    Each method has the signature of ``memforge.memory.store.MemoryStore``. A
    test that inspects search or vector state uses a real ``MemoryStore``.
    """

    def __init__(self, db: Database) -> None:
        self.db = db
        self.deleted: list[tuple[str, dict[str, Any]]] = []
        self.audit_events: list[tuple[str, str, dict[str, Any]]] = []

    async def delete_projected_document(
        self,
        doc_id: str,
        *,
        source_id: str,
        deletion_context: dict[str, Any] | None = None,
        source_activity: SourceActivityLease | None = None,
    ) -> None:
        self.deleted.append(
            (doc_id, {"source_id": source_id, "deletion_context": deletion_context, "source_activity": source_activity})
        )
        await self.db.delete_projected_document(doc_id, source_id=source_id, source_activity=source_activity)

    async def attempt_lifecycle_vector_delivery(
        self,
        lifecycle_plan_id: str | None = None,
        *,
        source_id: str | None = None,
    ) -> LifecycleVectorDeliveryResult:
        return LifecycleVectorDeliveryResult(
            state=LifecycleVectorDeliveryState.DELIVERED,
            attempted_tasks=0,
            failed_tasks=0,
            error_types=(),
        )

    async def record_audit_event(
        self,
        event_type: str,
        status: str,
        *,
        context: AuditContext | None = None,
        **fields: Any,
    ) -> None:
        self.audit_events.append((event_type, status, {"context": context, **fields}))

    def operation_context(self, **fields: str | None) -> AuditContext:
        return AuditContext(**fields)
