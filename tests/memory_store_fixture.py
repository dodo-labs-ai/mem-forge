"""The memory store a test sync orchestrator writes through."""

from __future__ import annotations

from typing import Any

from memforge.storage.database import Database


class OrchestratorMemoryStore:
    """The memory-store operations the sync orchestrator calls, against the relational store only.

    A test that inspects search or vector state uses a real ``MemoryStore``.
    """

    def __init__(self, db: Database | None = None) -> None:
        self.db = db
        self.deleted: list[tuple[str, dict[str, Any]]] = []
        self.audit_events: list[tuple[str, str, dict[str, Any]]] = []

    async def delete_projected_document(self, doc_id: str, **kwargs: Any) -> None:
        self.deleted.append((doc_id, kwargs))
        await self.db.delete_projected_document(
            doc_id,
            source_id=kwargs["source_id"],
            source_activity=kwargs.get("source_activity"),
        )

    async def attempt_lifecycle_vector_delivery(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def record_audit_event(self, event_type: str, status: str, **fields: Any) -> None:
        self.audit_events.append((event_type, status, fields))

    def operation_context(self, **fields: str | None) -> None:
        return None
