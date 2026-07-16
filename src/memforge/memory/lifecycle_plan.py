"""Complete, stale-guarded lifecycle plans for one reconciliation scope."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Mapping, Protocol, Sequence, runtime_checkable

from memforge.memory.evidence import (
    EvidenceReference,
    EvidenceUnit,
    validate_evidence_references,
)


class LifecycleGateState(str, Enum):
    GATED = "gated"
    ENABLED = "enabled"


class LifecyclePlanStatus(str, Enum):
    STAGED = "staged"
    APPLIED = "applied"
    REJECTED = "rejected"
    STALE = "stale"


class CutoverFindingStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"


class CutoverFindingReason(str, Enum):
    MISSING_SOURCE_PROVENANCE = "missing_source_provenance"
    OBSERVATION_NOT_FOUND = "observation_not_found"
    AMBIGUOUS_OBSERVATION = "ambiguous_observation"
    LINEAGE_VALIDATION_FAILED = "lineage_validation_failed"


class HistoricalProjectionFailureReason(str, Enum):
    """Deterministic absence reasons that may justify cutover retirement."""

    DOCUMENT_MISSING = "document_missing"
    RAW_ARTIFACT_MISSING = "raw_artifact_missing"
    NORMALIZED_ARTIFACT_MISSING = "normalized_artifact_missing"
    CANONICAL_CONCEPT_MISSING = "canonical_concept_missing"
    CANONICAL_CONCEPT_EMPTY = "canonical_concept_empty"
    EXACT_INPUTS_MISSING = "exact_inputs_missing"


AGENT_SESSION_TERMINAL_PROJECTION_FAILURES = frozenset(
    {
        HistoricalProjectionFailureReason.DOCUMENT_MISSING,
        HistoricalProjectionFailureReason.CANONICAL_CONCEPT_MISSING,
        HistoricalProjectionFailureReason.CANONICAL_CONCEPT_EMPTY,
        HistoricalProjectionFailureReason.EXACT_INPUTS_MISSING,
    }
)


def build_unprovable_cutover_resolution(
    *,
    reconstruction_attempt_id: str,
    operator_id: str,
    unavailable_documents: Mapping[str, str],
) -> dict[str, object]:
    """Normalize the exact terminal evidence persisted by both adapters."""

    if not reconstruction_attempt_id.strip() or not operator_id.strip():
        raise ValueError("unprovable retirement requires operator and reconstruction attempt ids")
    normalized: dict[str, str] = {}
    for document_id, raw_reason in unavailable_documents.items():
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("unprovable retirement requires deterministic unavailable documents")
        reason = HistoricalProjectionFailureReason(raw_reason)
        if reason not in AGENT_SESSION_TERMINAL_PROJECTION_FAILURES:
            raise ValueError("unprovable retirement requires exhausted Agent Session recovery paths")
        normalized[document_id.strip()] = reason.value
    if len(normalized) != len(unavailable_documents) or not normalized:
        raise ValueError("unprovable retirement requires deterministic unavailable documents")
    return {
        "kind": "unprovable_source_retired",
        "operator_id": operator_id.strip(),
        "reconstruction_attempt_id": reconstruction_attempt_id.strip(),
        "unavailable_documents": dict(sorted(normalized.items())),
    }


def validate_unprovable_cutover_evidence(
    *,
    available_provenance: Mapping[str, object],
    mapping_attempt: Mapping[str, object],
    source_rows: Sequence[Mapping[str, object]],
    source_id: str,
    unavailable_documents: Mapping[str, str],
) -> tuple[str, ...]:
    """Reject every malformed or contradictory entry before destructive cutover."""

    raw_documents = available_provenance.get("documents")
    raw_attempts = mapping_attempt.get("attempts")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("unprovable retirement requires strict exact source provenance")
    if not isinstance(raw_attempts, list) or not raw_attempts or not source_rows:
        raise ValueError("unprovable retirement requires strict exact source provenance")

    document_ids: list[str] = []
    for item in raw_documents:
        if not isinstance(item, Mapping) or set(item) != {"doc_id", "source_type", "excerpt"}:
            raise ValueError("unprovable retirement requires strict exact source provenance")
        doc_id = item.get("doc_id")
        excerpt = item.get("excerpt")
        if (
            not isinstance(doc_id, str)
            or not doc_id.strip()
            or item.get("source_type") != "agent_session"
            or (excerpt is not None and not isinstance(excerpt, str))
        ):
            raise ValueError("unprovable retirement requires strict exact source provenance")
        document_ids.append(doc_id.strip())

    attempt_ids: list[str] = []
    for item in raw_attempts:
        if not isinstance(item, Mapping) or set(item) != {"doc_id", "result"}:
            raise ValueError("unprovable retirement requires strict exact source provenance")
        doc_id = item.get("doc_id")
        if (
            not isinstance(doc_id, str)
            or not doc_id.strip()
            or item.get("result") != "source_unit_not_found"
        ):
            raise ValueError("unprovable retirement requires strict exact source provenance")
        attempt_ids.append(doc_id.strip())

    edge_ids: list[str] = []
    for row in source_rows:
        if set(row) != {"doc_id", "source_id", "source_type"}:
            raise ValueError("unprovable retirement requires strict exact source provenance")
        doc_id = row.get("doc_id")
        if (
            not isinstance(doc_id, str)
            or not doc_id.strip()
            or row.get("source_id") != source_id
            or row.get("source_type") != "agent_session"
        ):
            raise ValueError("unprovable retirement requires exclusive source provenance")
        edge_ids.append(doc_id.strip())

    unavailable_ids = list(unavailable_documents)
    if (
        len(set(document_ids)) != len(document_ids)
        or len(set(attempt_ids)) != len(attempt_ids)
        or len(set(edge_ids)) != len(edge_ids)
        or set(document_ids) != set(attempt_ids)
        or set(document_ids) != set(edge_ids)
        or set(document_ids) != set(unavailable_ids)
    ):
        raise ValueError("unprovable retirement requires strict exact source provenance")
    return tuple(sorted(document_ids))


def unprovable_cutover_retirement_plan_id(finding_id: str) -> str:
    """Stable lifecycle-plan identity for one terminal cutover finding."""

    digest = sha256(f"unprovable-cutover-retirement\x1f{finding_id}".encode()).hexdigest()[:20]
    return f"lifecycle-cutover-retire-{digest}"


class LifecycleBackfillJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class LifecycleVectorOperation(str, Enum):
    UPSERT = "upsert"
    DELETE = "delete"


class LifecycleVectorTaskStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class LifecycleReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class LifecycleGate:
    source_id: str
    state: LifecycleGateState
    reason: str | None = None
    enabled_at: str | None = None
    audited_at: str | None = None


@dataclass(frozen=True, slots=True)
class LifecycleCutoverFinding:
    id: str
    source_id: str
    memory_id: str
    reason: CutoverFindingReason
    status: CutoverFindingStatus
    available_provenance: Mapping[str, object]
    mapping_attempt: Mapping[str, object]
    observation_id: str | None = None
    source_unit_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    resolved_at: str | None = None


@dataclass(frozen=True, slots=True)
class LifecycleBackfillJob:
    id: str
    source_id: str
    status: LifecycleBackfillJobStatus
    scanned_memories: int = 0
    mapped_memories: int = 0
    finding_count: int = 0
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True, slots=True)
class LegacyMemoryProvenance:
    memory_id: str
    doc_id: str
    source_id: str
    source_type: str
    content: str
    excerpt: str | None
    visibility: str
    owner_user_id: str | None
    project_key: str | None
    repo_identifier: str | None


@dataclass(frozen=True, slots=True)
class LifecycleVectorTask:
    id: str
    lifecycle_plan_id: str
    memory_id: str
    operation: LifecycleVectorOperation
    status: LifecycleVectorTaskStatus
    attempts: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LifecycleReview:
    id: str
    lifecycle_plan_id: str
    incumbent_memory_id: str
    status: LifecycleReviewStatus
    staged_evidence: Mapping[str, object]
    reason: str | None = None
    created_at: str | None = None
    resolved_at: str | None = None


class IncumbentDisposition(str, Enum):
    KEEP = "keep"
    REMOVE_SUPPORT = "remove_support"
    SUPERSEDE = "supersede"
    REVIEW = "review"


class LifecycleMutationType(str, Enum):
    CREATE_MEMORY = "create_memory"
    ATTACH_SUPPORT = "attach_support"
    REMOVE_SUPPORT = "remove_support"
    SUPERSEDE_MEMORY = "supersede_memory"
    RETIRE_MEMORY = "retire_memory"
    CREATE_REVIEW = "create_review"
    RESOLVE_REVIEW = "resolve_review"
    REFRESH_MEMORY_INDEX = "refresh_memory_index"


DESTRUCTIVE_MUTATIONS = frozenset(
    {
        LifecycleMutationType.REMOVE_SUPPORT,
        LifecycleMutationType.SUPERSEDE_MEMORY,
        LifecycleMutationType.RETIRE_MEMORY,
    }
)


@dataclass(frozen=True, slots=True)
class ReconciliationScope:
    """Atomic planning boundary, normally one changed Source Unit."""

    id: str
    source_id: str
    source_unit_id: str
    base_unit_revision_id: str | None
    target_unit_revision_id: str | None
    dependency_unit_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IncumbentDecision:
    memory_id: str
    disposition: IncumbentDisposition
    reason: str
    replacement_memory_id: str | None = None

    def __post_init__(self) -> None:
        if self.disposition is IncumbentDisposition.SUPERSEDE and not self.replacement_memory_id:
            raise ValueError("supersede decision requires replacement_memory_id")


@dataclass(frozen=True, slots=True)
class CoverageProof:
    """Proof that bounded model batches closed the entire incumbent ledger."""

    mandatory_incumbent_ids: tuple[str, ...]
    incumbent_decisions: tuple[IncumbentDecision, ...]
    batch_ids: tuple[str, ...]
    completed_batch_ids: tuple[str, ...]

    def validate(self) -> None:
        expected = set(self.mandatory_incumbent_ids)
        if len(expected) != len(self.mandatory_incumbent_ids):
            raise ValueError("duplicate mandatory incumbent id")
        decision_ids = [item.memory_id for item in self.incumbent_decisions]
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("duplicate incumbent decision")
        missing = expected.difference(decision_ids)
        if missing:
            raise ValueError(f"missing incumbent decisions: {sorted(missing)}")
        extra = set(decision_ids).difference(expected)
        if extra:
            raise ValueError(f"unexpected incumbent decisions: {sorted(extra)}")
        if set(self.completed_batch_ids) != set(self.batch_ids):
            raise ValueError("incomplete reconciliation batches")


@dataclass(frozen=True, slots=True)
class StaleGuard:
    observation_revision_ids: tuple[str, ...]
    support_set_hashes: Mapping[str, str]
    memory_versions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LifecycleMutation:
    mutation_type: LifecycleMutationType
    memory_id: str
    source_id: str
    evidence_reference_ids: tuple[str, ...] = ()
    replacement_memory_id: str | None = None
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mutation_type is LifecycleMutationType.SUPERSEDE_MEMORY and not self.replacement_memory_id:
            raise ValueError("supersede mutation requires replacement_memory_id")
        if self.mutation_type in {
            LifecycleMutationType.ATTACH_SUPPORT,
            LifecycleMutationType.REMOVE_SUPPORT,
        } and not self.evidence_reference_ids:
            raise ValueError("support mutation requires evidence_reference_ids")


@dataclass(frozen=True, slots=True)
class LifecyclePlan:
    id: str
    scope: ReconciliationScope
    gate_state: LifecycleGateState
    coverage_proof: CoverageProof
    stale_guard: StaleGuard
    mutations: tuple[LifecycleMutation, ...]
    evidence_units: tuple[EvidenceUnit, ...] = ()
    evidence_references: tuple[EvidenceReference, ...] = ()

    def validate(self) -> None:
        self.coverage_proof.validate()
        if any(item.source_id != self.scope.source_id for item in self.mutations):
            raise ValueError("plan mutation belongs to another source")
        if self.gate_state is LifecycleGateState.GATED and any(
            item.mutation_type in DESTRUCTIVE_MUTATIONS for item in self.mutations
        ):
            raise ValueError("destructive mutation rejected by lifecycle gate")
        incumbents = set(self.coverage_proof.mandatory_incumbent_ids)
        for item in self.mutations:
            if item.mutation_type in DESTRUCTIVE_MUTATIONS and item.memory_id not in incumbents:
                raise ValueError("destructive mutation targets memory outside mandatory incumbent ledger")
        unit_ids = {item.id for item in self.evidence_units}
        if len(unit_ids) != len(self.evidence_units):
            raise ValueError("duplicate staged Evidence Unit")
        if any(item.source_id != self.scope.source_id for item in self.evidence_units):
            raise ValueError("staged Evidence Unit belongs to another source")
        for reference in self.evidence_references:
            if not reference.id or reference.evidence_unit_id not in unit_ids:
                raise ValueError("staged Evidence Reference requires a staged Evidence Unit and stable id")
        references_by_unit = {
            unit_id: tuple(
                reference
                for reference in self.evidence_references
                if reference.evidence_unit_id == unit_id
            )
            for unit_id in unit_ids
        }
        available_revisions = set(self.stale_guard.observation_revision_ids)
        for unit_id, references in references_by_unit.items():
            if not references:
                raise ValueError(f"staged Evidence Unit lacks references: {unit_id}")
            validate_evidence_references(
                references,
                available_revision_ids=available_revisions,
            )


@runtime_checkable
class LifecyclePlanStore(Protocol):
    """Storage boundary that validates stale guards and commits atomically."""

    async def apply_lifecycle_plan(self, plan: LifecyclePlan) -> None: ...


def lifecycle_plan_to_payload(plan: LifecyclePlan) -> dict[str, object]:
    return {
        "id": plan.id,
        "scope": {
            "id": plan.scope.id,
            "source_id": plan.scope.source_id,
            "source_unit_id": plan.scope.source_unit_id,
            "base_unit_revision_id": plan.scope.base_unit_revision_id,
            "target_unit_revision_id": plan.scope.target_unit_revision_id,
            "dependency_unit_ids": list(plan.scope.dependency_unit_ids),
        },
        "gate_state": plan.gate_state.value,
        "coverage_proof": {
            "mandatory_incumbent_ids": list(plan.coverage_proof.mandatory_incumbent_ids),
            "incumbent_decisions": [
                {
                    "memory_id": item.memory_id,
                    "disposition": item.disposition.value,
                    "reason": item.reason,
                    "replacement_memory_id": item.replacement_memory_id,
                }
                for item in plan.coverage_proof.incumbent_decisions
            ],
            "batch_ids": list(plan.coverage_proof.batch_ids),
            "completed_batch_ids": list(plan.coverage_proof.completed_batch_ids),
        },
        "stale_guard": {
            "observation_revision_ids": list(plan.stale_guard.observation_revision_ids),
            "support_set_hashes": dict(plan.stale_guard.support_set_hashes),
            "memory_versions": dict(plan.stale_guard.memory_versions),
        },
        "evidence_units": [
            {
                "id": item.id,
                "source_id": item.source_id,
                "doc_id": item.doc_id,
                "doc_revision_id": item.doc_revision_id,
                "source_type": item.source_type,
                "source_anchor": item.source_anchor,
                "source_lineage_id": item.source_lineage_id,
                "project_key": item.project_key,
                "visibility": item.visibility,
                "owner_user_id": item.owner_user_id,
                "repo_identifier": item.repo_identifier,
                "content": item.content,
                "excerpt": item.excerpt,
                "evidence_provenance": item.evidence_provenance.value,
                "client": item.client,
                "source_metadata": dict(item.source_metadata),
                "observed_at": item.observed_at,
                "extractor_run_id": item.extractor_run_id,
                "access_context_hash": item.access_context_hash,
            }
            for item in plan.evidence_units
        ],
        "evidence_references": [
            {
                "id": item.id,
                "evidence_unit_id": item.evidence_unit_id,
                "role": item.role.value,
                "anchor": {
                    "kind": item.anchor.kind.value,
                    "observation_id": item.anchor.observation_id,
                    "observation_revision_id": item.anchor.observation_revision_id,
                    "fragment_id": item.anchor.fragment_id,
                    "range_start": item.anchor.range_start,
                    "range_end": item.anchor.range_end,
                },
            }
            for item in plan.evidence_references
        ],
        "mutations": [
            {
                "mutation_type": item.mutation_type.value,
                "memory_id": item.memory_id,
                "source_id": item.source_id,
                "evidence_reference_ids": list(item.evidence_reference_ids),
                "replacement_memory_id": item.replacement_memory_id,
                "payload": dict(item.payload),
            }
            for item in plan.mutations
        ],
    }
