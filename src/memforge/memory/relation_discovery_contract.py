"""Durable, provider-neutral contract for progressive Memory relation discovery."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256


CURRENT_RELATION_EVIDENCE_PREDICATE_SQL = """
msa.memory_id = ? AND msa.source_id = ? AND msa.active = 1
AND er.role = 'primary'
AND eu.source_id = ? AND eu.source_lineage_id = ?
AND so.source_id = eu.source_id
AND so.source_unit_id = eu.source_lineage_id
AND er.observation_revision_id = so.current_revision_id
""".strip()


# Stored failure text is bounded; the error code carries the stable classification.
RELATION_DISCOVERY_ERROR_MAX_CHARS = 4000
# Audit event recorded for each work item an operator re-runs, before it is reset.
RELATION_DISCOVERY_RERUN_EVENT = "relation_discovery_rerun"


class RelationDiscoveryWorkStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    OBSOLETE = "obsolete"


class RelationDiscoveryWorkState(str, Enum):
    """Operator-facing states; exhausted is failed work that used every attempt."""

    EXHAUSTED = "exhausted"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RelationDiscoveryWorkSelection:
    """Which durable discovery work an operator lists, counts or re-runs."""

    state: RelationDiscoveryWorkState
    max_attempts: int
    error_code: str | None = None
    # Half-open range over the work's last update; a time without a zone is UTC.
    updated_from: datetime | None = None
    updated_to: datetime | None = None
    classifier_version: str | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("relation discovery selection requires positive max attempts")

    @property
    def rerunnable(self) -> bool:
        """Only finished work is re-run; failed work still retrying is left to its schedule."""

        return self.state is not RelationDiscoveryWorkState.FAILED


def stored_work_time(moment: datetime) -> str:
    """A time in the form work times are stored: ISO 8601 in UTC, so text order is time order."""

    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat()


def relation_discovery_work_selection_sql(
    selection: RelationDiscoveryWorkSelection,
) -> tuple[str, tuple[object, ...]]:
    """Portable predicate over relation_discovery_work for one selection."""

    clauses: list[str]
    params: list[object]
    if selection.state is RelationDiscoveryWorkState.EXHAUSTED:
        clauses, params = ["status = 'failed'", "attempts >= ?"], [selection.max_attempts]
    elif selection.state is RelationDiscoveryWorkState.FAILED:
        clauses, params = ["status = 'failed'", "attempts < ?"], [selection.max_attempts]
    else:
        clauses, params = ["status = 'completed'"], []
    for column, operator, value in (
        ("error_code", "=", selection.error_code),
        ("updated_at", ">=", stored_work_time(selection.updated_from) if selection.updated_from else None),
        ("updated_at", "<", stored_work_time(selection.updated_to) if selection.updated_to else None),
        ("classifier_version", "=", selection.classifier_version),
    ):
        if value is not None:
            clauses.append(f"{column} {operator} ?")
            params.append(value)
    return " AND ".join(clauses), tuple(params)


def rerunnable_relation_discovery_work_sql(max_attempts: int) -> tuple[str, tuple[object, ...]]:
    """Portable predicate for work an operator may re-run: completed or exhausted."""

    completed, completed_params = relation_discovery_work_selection_sql(
        RelationDiscoveryWorkSelection(state=RelationDiscoveryWorkState.COMPLETED, max_attempts=max_attempts)
    )
    exhausted, exhausted_params = relation_discovery_work_selection_sql(
        RelationDiscoveryWorkSelection(state=RelationDiscoveryWorkState.EXHAUSTED, max_attempts=max_attempts)
    )
    return f"(({completed}) OR ({exhausted}))", (*completed_params, *exhausted_params)


def relation_discovery_work_rerunnable(work: RelationDiscoveryWork, *, max_attempts: int) -> bool:
    """The same rule as ``rerunnable_relation_discovery_work_sql``, for one loaded item."""

    return work.status is RelationDiscoveryWorkStatus.COMPLETED or (
        work.status is RelationDiscoveryWorkStatus.FAILED and work.attempts >= max_attempts
    )


def resolve_relation_discovery_actor_user_id(
    *,
    visibility: str,
    owner_user_id: str | None,
    requested_actor_user_id: str | None,
) -> str | None:
    """Resolve the principal that owns one durable relation-discovery decision."""

    if visibility == "private":
        if not owner_user_id:
            raise ValueError("private relation discovery requires owner identity")
        return owner_user_id
    return requested_actor_user_id


@dataclass(frozen=True, slots=True)
class RelationDiscoveryRequest:
    """One bounded post-commit discovery request for an activated Memory."""

    id: str
    memory_id: str
    expected_content_hash: str
    source_id: str
    source_unit_id: str
    # The Source Unit revision whose commit queued the request, kept for audit;
    # completion is fenced by the challenger's content and current evidence.
    source_unit_revision_id: str | None
    doc_id: str
    actor_user_id: str | None
    entity_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        required = {
            "id": self.id,
            "memory_id": self.memory_id,
            "expected_content_hash": self.expected_content_hash,
            "source_id": self.source_id,
            "source_unit_id": self.source_unit_id,
            "doc_id": self.doc_id,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise ValueError("relation discovery request requires " + ", ".join(missing))


@dataclass(frozen=True, slots=True)
class RelationDiscoveryWork:
    """A leased durable request; attempts are fenced by worker and lease token.

    ``run_generation`` counts operator re-runs, so each run of the same request
    records a distinct relation run. ``error`` and ``error_code`` describe the
    last failure; ``classifier_version`` names the classifier of the last
    completed run.
    """

    request: RelationDiscoveryRequest
    lifecycle_plan_id: str
    status: RelationDiscoveryWorkStatus
    attempts: int = 0
    lease_owner: str | None = None
    lease_token: str | None = None
    lease_until: str | None = None
    next_attempt_at: str | None = None
    error: str | None = None
    error_code: str | None = None
    run_generation: int = 0
    classifier_version: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None


def relation_discovery_request_id(
    *,
    lifecycle_plan_id: str,
    memory_id: str,
    expected_content_hash: str,
) -> str:
    digest = sha256("\x1f".join((lifecycle_plan_id, memory_id, expected_content_hash)).encode("utf-8")).hexdigest()[:20]
    return f"relation-work-{digest}"
