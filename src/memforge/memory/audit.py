"""Append-only audit events for memory evaluation.

The audit ledger records memory decisions and storage/index outcomes. It is
not a source of truth for memory state; SQLite memory tables remain the source
of truth. Audit rows give evaluators enough context to understand why state
changed and whether related side effects succeeded.
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = [
    "AuditContext",
    "MemoryAuditEvent",
    "MemoryAuditLogger",
    "generate_audit_event_id",
    "generate_operation_id",
]


def generate_audit_event_id() -> str:
    return f"evt-{uuid.uuid4().hex[:12]}"


def generate_operation_id() -> str:
    return f"op-{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AuditContext:
    """Context propagated from entry points into memory mutation boundaries."""

    run_id: str | None = None
    trace_id: str | None = None
    operation_id: str | None = None
    actor_type: str | None = None
    actor_id: str | None = None
    source_id: str | None = None
    doc_id: str | None = None
    model: str | None = None
    prompt_hash: str | None = None
    config_hash: str | None = None

    def child(self, **overrides: str | None) -> "AuditContext":
        values = {k: v for k, v in overrides.items() if v is not None}
        if "operation_id" not in values:
            values["operation_id"] = generate_operation_id()
        return replace(self, **values)


@dataclass
class MemoryAuditEvent:
    """One append-only memory audit event."""

    event_type: str
    status: str
    event_id: str = field(default_factory=generate_audit_event_id)
    operation_id: str | None = None
    parent_event_id: str | None = None
    occurred_at: datetime | None = None
    actor_type: str | None = None
    actor_id: str | None = None
    run_id: str | None = None
    trace_id: str | None = None
    source_id: str | None = None
    doc_id: str | None = None
    memory_id: str | None = None
    candidate_id: str | None = None
    review_id: str | None = None
    support_kind: str | None = None
    decision: str | None = None
    reason: str | None = None
    payload_class: str | None = None
    before_snapshot: dict[str, Any] | None = None
    after_snapshot: dict[str, Any] | None = None
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    prompt_hash: str | None = None
    config_hash: str | None = None
    thresholds: dict[str, Any] | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.occurred_at is None:
            self.occurred_at = _now()
        if self.operation_id is None:
            self.operation_id = generate_operation_id()


class MemoryAuditLogger:
    """Writes audit events with default context applied."""

    def __init__(self, db: "Database", default_context: AuditContext | None = None) -> None:
        self.db = db
        self.default_context = default_context or AuditContext()

    async def emit(
        self,
        event_type: str,
        status: str,
        *,
        context: AuditContext | None = None,
        **fields: Any,
    ) -> MemoryAuditEvent:
        ctx = context or self.default_context
        event = MemoryAuditEvent(
            event_type=event_type,
            status=status,
            operation_id=fields.pop("operation_id", None) or ctx.operation_id,
            actor_type=fields.pop("actor_type", None) or ctx.actor_type,
            actor_id=fields.pop("actor_id", None) or ctx.actor_id,
            run_id=fields.pop("run_id", None) or ctx.run_id,
            trace_id=fields.pop("trace_id", None) or ctx.trace_id,
            source_id=fields.pop("source_id", None) or ctx.source_id,
            doc_id=fields.pop("doc_id", None) or ctx.doc_id,
            model=fields.pop("model", None) or ctx.model,
            prompt_hash=fields.pop("prompt_hash", None) or ctx.prompt_hash,
            config_hash=fields.pop("config_hash", None) or ctx.config_hash,
            **fields,
        )
        try:
            await self.db.insert_memory_audit_event(event)
        except Exception:
            logger.exception("Memory audit event write failed: %s", event_type)
        return event
