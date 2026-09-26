"""Memory extraction and projected-lifecycle orchestration.

The engine is the ownership boundary between source projection/extraction and
durable memory state.  A projected lifecycle call derives its reconciliation
scope, access context, staged evidence, lifecycle plan, and outbox work from one
``SourceProjection`` so those records commit atomically.
"""

from __future__ import annotations

import asyncio
import json
import hashlib

from memforge.llm.structured import StructuredLlmError
import logging
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar

from memforge.evals.agent_evaluation import (
    AgentRuntimeBundle,
    NoOpRuntimeEventTraceSink,
    RuntimeEventTraceSink,
    assessment_sink_for_runtime_sink,
    bind_source_lifecycle_outcome,
    source_lifecycle_execution_identity,
    current_deployment_revision,
    publish_agent_assessments,
    publish_runtime_events,
)
from memforge.memory.candidate_admission import (
    CANDIDATE_ADMISSION_CONTRACT,
    CandidateAdmissionError,
    CandidateRejection,
    admit_candidates,
)
from memforge.memory.coordinator_review import (
    CarriedConflict,
    carried_conflicts,
    carry_into_ledger,
    is_coordinator_review,
)
from memforge.memory.entity_resolver import EntityResolver
from memforge.memory.evidence import (
    EvidenceReference,
    EvidenceUnit,
)
from memforge.memory.identity_resolver import (
    IdentityResolutionRequest,
    IdentityResolver,
)
from memforge.memory.lifecycle_plan import (
    LifecycleGateState,
    LifecyclePlan,
    LifecycleReview,
    LifecycleReviewStatus,
    ProjectedLifecycleDeferredError,
    AuthorityPlanStaleError,
    ProjectedSupportInvariantError,
    ReconciliationScope,
)
from memforge.memory.lifecycle_planner import (
    NewMemoryDefaults,
    build_lifecycle_plan,
    lifecycle_access_context_hash,
    lifecycle_memory_version,
    lifecycle_plan_id,
)
from memforge.memory.quality import classify_memory_candidate
from memforge.pipeline.projection_fragments import (
    SupportRevalidationLimitation,
    SupportRevalidationLimitationCode,
)
from memforge.pipeline.revision_assessment import RevisionAssessmentContext, SupportAssessment
from memforge.pipeline.support_relation_coordinator import (
    MemorySupport,
    RecheckReading,
    SupportRecheckRequest,
    memory_support,
)
from memforge.memory.cross_document_relation import StructuredCrossDocumentRelationClassifier
from memforge.memory.relation_candidate_retrieval import CrossDocumentCandidateRetriever
from memforge.memory.sparse_relation_classifier import (
    SPARSE_MEMORY_CLASSIFIER_VERSION,
    SparseMemoryRelationClassifier,
)
from memforge.memory.relation_classifier import MemoryPairClassificationError
from memforge.source_access import (
    memory_visibility_for_document,
    memory_visibility_for_source_id,
)
from memforge.source_activity import SourceActivityLease
from memforge.source_projection import ImpactResult, ProjectionCoverage, SourceUnitRevision, resolve_anchor_impact
from memforge.source_derivation import (
    SourceUnitDerivationContext,
    source_derivation_context_identity_hash,
    source_derivation_projection_identity_hash,
)
from memforge.storage.adapters.protocols import EntityResolutionScope
from memforge.llm.failure_trace import failure_trace_context, enrich_failure_trace_context
from memforge.models import (
    CoordinatorProposal,
    Memory,
    RawMemory,
    ReconcileAction,
    ReconcileOperation,
    content_hash,
    generate_memory_id,
    parse_memory_validity_date,
)

if TYPE_CHECKING:
    from memforge.memory.store import MemoryStore
    from memforge.models import DocumentRecord
    from memforge.source_projection import SourceProjection
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

_First = TypeVar("_First")
_Second = TypeVar("_Second")

__all__ = [
    "DeferredProjectedLifecycleHandle",
    "MemoryEngine",
    "SourceUnitLifecycleDeferred",
    "SourceUnitLifecycleExecutionError",
]


class SourceUnitLifecycleExecutionError(RuntimeError):
    """A failed lifecycle execution carrying its content-free terminal bundle."""

    def __init__(
        self,
        message: str,
        runtime_bundle: AgentRuntimeBundle,
        *,
        retryable: bool = True,
        commit_attempted: bool = True,
    ) -> None:
        super().__init__(message)
        self.runtime_bundle = runtime_bundle
        self.retryable = retryable
        self.commit_attempted = commit_attempted


@dataclass(frozen=True, slots=True)
class _PreparedLifecyclePlanInputs:
    plan_id: str
    scope: ReconciliationScope
    gate_state: LifecycleGateState
    operations: tuple[ReconcileOperation, ...]
    incumbents: Mapping[str, Memory]
    memory_authority_hashes: Mapping[str, str]
    initial_support_owners: Mapping[str, Mapping[str, str]]
    observation_revision_ids: tuple[str, ...]
    evidence_unit_ids_by_claim_hash: Mapping[str, tuple[str, ...]]
    corroboration_targets_by_claim_hash: Mapping[str, Memory]
    corroboration_proofs_by_claim_hash: Mapping[str, Mapping[str, object]]
    defaults: NewMemoryDefaults
    evidence_units: tuple[EvidenceUnit, ...]
    evidence_references: tuple[EvidenceReference, ...]


@dataclass(slots=True)
class _PreparedProjectedLifecycleCommit:
    """Opaque same-process semantic result for deterministic commit replay."""

    projection: "SourceProjection"
    plan_inputs: _PreparedLifecyclePlanInputs
    document: "DocumentRecord | None"
    derivation_id: str | None
    derivation_context_identity_hash: str | None
    required_derivation_work_ids: tuple[str, ...]
    expected_source_activity_epoch: int | None
    source_activity: SourceActivityLease | None
    base_stats: Mapping[str, int]
    corroboration_target_ids: frozenset[str]
    lifecycle_execution_owner_id: str | None
    operation_input_hash: str
    doc_id: str
    source_type: str
    started_at: float
    incumbent_count: int
    relation_pair_count: int
    model_call_count: int
    prepared_at_attempt_count: int
    admission_rejections: tuple[CandidateRejection, ...] = ()
    retry_attempt_count: int = 0
    applied_stats: dict[str, int] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    allowed_blocker_source_unit_ids: set[str] = field(
        default_factory=set,
        repr=False,
        compare=False,
    )

    @property
    def source_unit_id(self) -> str:
        return self.plan_inputs.scope.source_unit_id


class SourceUnitLifecycleDeferred(SourceUnitLifecycleExecutionError):
    """A non-terminal same-run commit conflict with an opaque prepared intent."""

    def __init__(
        self,
        message: str,
        runtime_bundle: AgentRuntimeBundle,
        *,
        handle: "DeferredProjectedLifecycleHandle",
    ) -> None:
        super().__init__(message, runtime_bundle, retryable=False)
        self.handle = handle


@dataclass(frozen=True, slots=True)
class DeferredProjectedLifecycleHandle:
    """Opaque same-process handle for one deferred projected commit."""

    source_unit_id: str
    blocking_source_unit_ids: tuple[str, ...]
    _prepared: _PreparedProjectedLifecycleCommit = field(repr=False)
    _runtime_bundle: AgentRuntimeBundle = field(repr=False)


def _prepared_memory_authority_hash(memory: Memory) -> str:
    """Hash Memory facts that semantic preparation was authorized to consume."""

    payload = {
        "id": memory.id,
        "memory_type": memory.memory_type,
        "content_hash": memory.content_hash,
        "visibility": memory.visibility,
        "owner_user_id": memory.owner_user_id,
        "project_key": memory.project_key,
        "repo_identifier": memory.repo_identifier,
        "confidence": memory.confidence,
        "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
        "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
        "status": memory.status,
        "superseded_by": memory.superseded_by,
        "retirement_reason": memory.retirement_reason,
        "replacement_kind": (
            memory.replacement_kind.value
            if memory.replacement_kind is not None
            else None
        ),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(slots=True)
class _LifecycleExecutionContext:
    started_at: float
    stage: str = "preparation"
    operation_input_hash: str | None = None
    incumbent_count: int = 0
    relation_pair_count: int = 0
    model_call_count: int = 0


class MemoryEngine:
    """Turns enrichment and extracted claims into durable memory state.

    Responsibilities:
    - Resolve extracted entities and aliases.
    - Build and apply an atomic lifecycle plan for a source projection.
    - Support direct, non-projected memory ingestion where explicitly requested.
    """

    def __init__(
        self,
        cross_document_candidates: CrossDocumentCandidateRetriever,
        db: Database,
        memory_store: MemoryStore,
        embed_cfg: dict | None = None,
        structured_llm_client: Any = None,
        llm_model: str = "claude-sonnet-4-20250514",
        runtime_event_trace_sink: RuntimeEventTraceSink | None = None,
        document_store: Any = None,
    ) -> None:
        self.cross_document_candidates = cross_document_candidates
        self.db = db
        self.memory_store = memory_store
        self.document_store = document_store
        self.structured_llm_client = structured_llm_client
        self.llm_model = llm_model
        self.runtime_event_trace_sink = runtime_event_trace_sink or NoOpRuntimeEventTraceSink()
        self.agent_assessment_sink = assessment_sink_for_runtime_sink(
            self.runtime_event_trace_sink
        )
        self.pair_classifier = (
            StructuredCrossDocumentRelationClassifier(
                client=structured_llm_client,
                model=llm_model,
            )
            if callable(getattr(structured_llm_client, "classify_cross_document_relations", None))
            else None
        )
        self.identity_resolver = IdentityResolver(
            memory_store=memory_store,
            pair_classifier=(
                SparseMemoryRelationClassifier(
                    client=structured_llm_client,
                    model=llm_model,
                )
                if callable(getattr(structured_llm_client, "discover_memory_relations", None))
                else None
            ),
            llm_model=llm_model,
        )
        # Entity resolver with embedding + LLM capabilities
        self.entity_resolver = EntityResolver(
            store=db,
            embed_cfg=embed_cfg,
            structured_llm_client=structured_llm_client,
            llm_model=llm_model,
        )

    async def _active_projected_incumbents(
        self,
        *,
        doc_id: str,
        source_unit_id: str,
    ) -> tuple[list[Memory], dict[str, tuple[str, ...]]]:
        """Load the complete active ledger by stable Unit, plus same-document provenance.

        A provider-backed rename can change ``doc_id`` without changing the
        Source Unit. Unit Support is therefore authoritative for the projected
        path; an active Memory with same-document extracted provenance but no
        Support in this Unit is included so the conservative lineage gate
        still sees it.
        """
        unit_support = await self.db.get_source_unit_support_unit_ids(source_unit_id)
        incumbents_by_id = {
            memory.id: memory for memory in await self.db.list_active_memories(tuple(sorted(unit_support)))
        }
        for memory in await self.db.get_memories_by_source_doc(
            doc_id,
            support_kind="extracted",
        ):
            if memory.status == "active":
                incumbents_by_id.setdefault(memory.id, memory)
        return [incumbents_by_id[key] for key in sorted(incumbents_by_id)], unit_support

    async def _projected_incumbent_impacts(
        self,
        *,
        projection: SourceProjection,
        incumbent_ids: frozenset[str],
        unit_support: Mapping[str, tuple[str, ...]],
    ) -> dict[str, ImpactResult]:
        """Resolve each incumbent against the current Revision Delta.

        Missing Unit Support, ambiguous mappings, and mixed evidence stay
        UNKNOWN. A single affected reference makes the incumbent AFFECTED;
        only a complete set of disjoint references proves DISJOINT.
        """

        delta = projection.deltas[0]
        ordered_incumbent_ids = tuple(sorted(incumbent_ids))
        evidence_by_memory_id = await self.db.get_active_memory_support_evidence_many(
            ordered_incumbent_ids,
            source_id=projection.source_id,
        )
        resolved: dict[str, ImpactResult] = {}
        for memory_id in ordered_incumbent_ids:
            unit_ids = frozenset(unit_support.get(memory_id, ()))
            if not unit_ids:
                resolved[memory_id] = ImpactResult.UNKNOWN
                continue
            evidence = evidence_by_memory_id.get(memory_id, ())
            impacts = {
                resolve_anchor_impact(item.anchor, delta)
                for item in evidence
                if item.evidence_unit_id in unit_ids
            }
            if ImpactResult.AFFECTED in impacts:
                resolved[memory_id] = ImpactResult.AFFECTED
            elif not impacts or ImpactResult.UNKNOWN in impacts:
                resolved[memory_id] = ImpactResult.UNKNOWN
            else:
                resolved[memory_id] = ImpactResult.DISJOINT
        return resolved

    @staticmethod
    def _partial_projection_protected_incumbents(
        *,
        projection: SourceProjection,
        incumbent_impacts: Mapping[str, ImpactResult],
    ) -> frozenset[str]:
        """Return partial-projection incumbents without affected-anchor proof."""

        if projection.coverage is not ProjectionCoverage.PARTIAL_PROJECTION:
            return frozenset()
        return frozenset(
            memory_id for memory_id, impact in incumbent_impacts.items() if impact is not ImpactResult.AFFECTED
        )

    @staticmethod
    def _enforce_partial_projection_keep(
        operations: tuple[ReconcileOperation, ...],
        protected_memory_ids: frozenset[str],
    ) -> tuple[ReconcileOperation, ...]:
        """Keep unproven incumbents while preserving non-destructive candidates."""

        protected: list[ReconcileOperation] = []
        for operation in operations:
            if operation.memory_id not in protected_memory_ids:
                protected.append(operation)
                continue
            if operation.action is ReconcileAction.UPDATE and operation.memory is not None:
                protected.append(
                    ReconcileOperation(
                        action=ReconcileAction.ADD,
                        memory=operation.memory,
                        reason="partial projection preserves candidate without mutating unproven incumbent",
                    )
                )
                protected.append(
                    ReconcileOperation(
                        action=ReconcileAction.NOOP,
                        memory_id=operation.memory_id,
                        reason="partial projection has no deterministic affected-anchor proof",
                    )
                )
                continue
            if operation.action is ReconcileAction.SUPERSEDE:
                protected.append(
                    replace(
                        operation,
                        reason="partial projection contradiction requires lifecycle review",
                        flag_for_review=True,
                    )
                )
                continue
            if operation.action is ReconcileAction.DELETE:
                protected.append(
                    ReconcileOperation(
                        action=ReconcileAction.NOOP,
                        memory_id=operation.memory_id,
                        reason="partial projection has no deterministic affected-anchor proof",
                    )
                )
                continue
            protected.append(operation)
        return tuple(protected)

    async def _carried_conflicts(
        self,
        *,
        source_unit_id: str,
        model_incumbents: Sequence[Memory],
        context: RevisionAssessmentContext | None,
    ) -> tuple[CarriedConflict, ...]:
        """This Source Unit's pending coordinator conflicts whose staged Candidate is still exactly current."""

        if context is None:
            return ()
        reviews = [
            review
            for review in await self.db.list_lifecycle_reviews(
                status=LifecycleReviewStatus.PENDING,
                incumbent_memory_ids=tuple(memory.id for memory in model_incumbents),
            )
            if is_coordinator_review(review, source_unit_id=source_unit_id)
        ]
        if not reviews:
            return ()
        return carried_conflicts(reviews, context.catalog(context.full_fragments))

    async def _coordinator_reviews(
        self, *, source_unit_id: str, incumbent_ids: Sequence[str],
    ) -> tuple[LifecycleReview, ...]:
        """Every coordinator Review this Source Unit raised on these incumbents, in any status."""

        return tuple(
            review
            for review in await self.db.list_lifecycle_reviews(incumbent_memory_ids=tuple(incumbent_ids))
            if is_coordinator_review(review, source_unit_id=source_unit_id)
        )

    async def _derivation_protected_incumbents(
        self,
        *,
        source_id: str,
        incumbent_ids: frozenset[str],
        protected_source_observation_ids: frozenset[str],
    ) -> frozenset[str]:
        """Keep incumbents whose current Support could not be re-derived."""

        if not incumbent_ids or not protected_source_observation_ids:
            return frozenset()
        observation_ids_by_memory_id = await self.db.get_active_memory_support_observation_ids_many(
            tuple(sorted(incumbent_ids)),
            source_id=source_id,
        )
        return frozenset(
            memory_id
            for memory_id, observation_ids in observation_ids_by_memory_id.items()
            if protected_source_observation_ids.intersection(observation_ids)
        )

    async def prepare_and_commit_projected_lifecycle(
        self,
        *,
        projection: SourceProjection,
        doc_id: str,
        raw_memories: list[RawMemory],
        doc_type: str,
        project_key: str | None,
        repo_identifier: str | None,
        document_content: str,
        update_mode: str,
        changed_hunks: str | None,
        update_plan_stats: dict[str, Any] | None,
        source_updated_at: datetime | None,
        user_id: str | None = None,
        protected_source_observation_ids: tuple[str, ...] = (),
        document: DocumentRecord | None = None,
        derivation_id: str | None = None,
        derivation_reprocess_all_current_observations: bool = False,
        derivation_reprocess_operation_id: str | None = None,
        derivation_support_without_baseline: bool = False,
        expected_source_activity_epoch: int | None = None,
        source_activity: SourceActivityLease | None = None,
        current_changed_ranges: tuple[tuple[int, int], ...] = (),
        lifecycle_execution_owner_id: str | None = None,
        lifecycle_attempt_count: int = 1,
    ) -> dict[str, int]:
        """Observe one complete Source Unit lifecycle execution."""

        runtime_context = _LifecycleExecutionContext(started_at=perf_counter())
        try:
            with failure_trace_context(source_id=projection.source_id, source_type=projection.source_type,
                    doc_id=doc_id, projection_run_id=projection.run_id, derivation_id=derivation_id,
                    source_unit_id=projection.source_unit_revisions[0].source_unit_id,
                    target_unit_revision_id=projection.source_unit_revisions[0].id,
                    execution_owner_id=lifecycle_execution_owner_id, lifecycle_attempt=lifecycle_attempt_count):
                return await self._prepare_and_commit_projected_lifecycle_once(
                    projection=projection,
                    doc_id=doc_id,
                    raw_memories=raw_memories,
                    doc_type=doc_type,
                    project_key=project_key,
                    repo_identifier=repo_identifier,
                    document_content=document_content,
                    update_mode=update_mode,
                    changed_hunks=changed_hunks,
                    update_plan_stats=update_plan_stats,
                    source_updated_at=source_updated_at,
                    user_id=user_id,
                    protected_source_observation_ids=protected_source_observation_ids,
                    document=document,
                    derivation_id=derivation_id,
                    derivation_reprocess_all_current_observations=(
                        derivation_reprocess_all_current_observations
                    ),
                    derivation_reprocess_operation_id=(
                        derivation_reprocess_operation_id
                    ),
                    derivation_support_without_baseline=derivation_support_without_baseline,
                    expected_source_activity_epoch=expected_source_activity_epoch,
                    source_activity=source_activity,
                    current_changed_ranges=current_changed_ranges,
                    lifecycle_execution_owner_id=lifecycle_execution_owner_id,
                    lifecycle_attempt_count=lifecycle_attempt_count,
                    _runtime_context=runtime_context,
                )
        except SourceUnitLifecycleExecutionError:
            raise
        except Exception as exc:
            if (
                lifecycle_execution_owner_id is None
                or runtime_context.operation_input_hash is None
                or len(projection.deltas) != 1
            ):
                raise
            delta = projection.deltas[0]
            from memforge.pipeline.reconciler import ReconciliationContractError

            support_limitation_reason = (
                {
                    SupportRevalidationLimitationCode.UNSUPPORTED_REPRESENTATION: (
                        "support_revalidation_unsupported_representation"
                    ),
                    SupportRevalidationLimitationCode.COMPILER_FAILURE: (
                        "support_revalidation_compiler_failure"
                    ),
                    SupportRevalidationLimitationCode.CAPACITY_EXCEEDED: (
                        "support_revalidation_capacity_exceeded"
                    ),
                }[exc.code]
                if isinstance(exc, SupportRevalidationLimitation)
                else None
            )
            reason_code = (
                "authority_plan_stale"
                if isinstance(exc, AuthorityPlanStaleError)
                else support_limitation_reason
                or (exc.reason_code if isinstance(exc, (ReconciliationContractError, CandidateAdmissionError)) else None)
                or {
                    "candidate_admission": "candidate_admission_failed",
                    "reconciliation": "reconciliation_failed",
                    "support_revalidation": "support_revalidation_failed",
                    "plan_construction": "lifecycle_plan_construction_failed",
                    "lifecycle_commit": "lifecycle_commit_failed",
                }.get(runtime_context.stage, "lifecycle_execution_failed")
            )
            bundle = bind_source_lifecycle_outcome(
                source_id=projection.source_id,
                source_type=projection.source_type,
                doc_id=doc_id,
                source_unit_id=delta.source_unit_id,
                base_unit_revision_id=delta.previous_unit_revision_id,
                target_unit_revision_id=delta.current_unit_revision_id,
                projection_run_id=projection.run_id,
                operation_input_hash=runtime_context.operation_input_hash,
                execution_owner_id=lifecycle_execution_owner_id,
                outcome="failed",
                reason_code=reason_code,
                attempt_count=lifecycle_attempt_count,
                duration_ms=max(
                    0,
                    round((perf_counter() - runtime_context.started_at) * 1000),
                ),
                incumbent_count=runtime_context.incumbent_count,
                relation_pair_count=runtime_context.relation_pair_count,
                mutation_count=0,
                review_count=0,
                model_call_count=runtime_context.model_call_count,
                deployment_revision=current_deployment_revision(),
                operation="assess_revision_support" if runtime_context.stage == "support_revalidation" else None,
                terminal_category=(
                    "invalid_response"
                    if isinstance(exc, ReconciliationContractError)
                    else (exc.terminal_category if isinstance(exc, (StructuredLlmError, CandidateAdmissionError)) else None)
                ),
                error_code=(
                    (exc.reason_code if isinstance(exc, (ReconciliationContractError, CandidateAdmissionError)) else
                          exc.error_code if isinstance(exc, StructuredLlmError) else None)
                ),
                validation_fields=getattr(exc, "validation_fields", ()),
                diagnostic=getattr(exc, "diagnostic", None),
            )
            raise SourceUnitLifecycleExecutionError(
                str(exc),
                bundle,
                retryable=not isinstance(
                    exc,
                    (
                        ProjectedSupportInvariantError,
                        SupportRevalidationLimitation,
                        ReconciliationContractError,
                    ),
                ) and getattr(exc, "retryable", True) and not (isinstance(exc, StructuredLlmError) and exc.terminal_category == "invalid_response"),
            ) from exc

    async def _materialize_prepared_projected_plan(
        self,
        prepared: _PreparedProjectedLifecycleCommit,
    ) -> LifecyclePlan:
        """Refresh only deterministic commit inputs for one prepared intent."""

        inputs = prepared.plan_inputs
        gate = await self.db.get_lifecycle_gate(inputs.scope.source_id)
        if gate.state is not inputs.gate_state:
            raise ProjectedSupportInvariantError(
                "prepared lifecycle gate changed before commit"
            )
        visibility, owner_user_id = await memory_visibility_for_source_id(
            self.db,
            source_id=inputs.scope.source_id,
        )
        current_access_hash = lifecycle_access_context_hash(
            visibility=visibility,
            owner_user_id=owner_user_id,
            project_key=inputs.defaults.project_key,
            repo_identifier=inputs.defaults.repo_identifier,
        )
        if current_access_hash != inputs.defaults.access_context_hash:
            raise ProjectedSupportInvariantError(
                "prepared lifecycle access context changed before commit"
            )

        memory_ids = tuple(sorted(inputs.memory_authority_hashes))
        current_memories = {
            memory.id: memory
            for memory in await self.db.list_memories_by_ids(memory_ids)
        }
        if set(current_memories) != set(memory_ids):
            raise ProjectedSupportInvariantError(
                "prepared lifecycle Memory set changed before commit"
            )
        for memory_id, expected_hash in inputs.memory_authority_hashes.items():
            if (
                _prepared_memory_authority_hash(current_memories[memory_id])
                != expected_hash
            ):
                raise ProjectedSupportInvariantError(
                    f"prepared lifecycle Memory changed before commit: {memory_id}"
                )

        current_support_owners = await self._active_support_owners(
            memory_ids
        )
        changed_owner_units: set[str] = set()
        for memory_id in memory_ids:
            initial = dict(inputs.initial_support_owners.get(memory_id, {}))
            current = dict(current_support_owners.get(memory_id, {}))
            shared_ids = set(initial).intersection(current)
            if any(initial[unit_id] != current[unit_id] for unit_id in shared_ids):
                raise ProjectedSupportInvariantError(
                    "prepared lifecycle Support ownership changed before commit"
                )
            changed_owner_units.update(
                initial[unit_id]
                for unit_id in set(initial).difference(current)
            )
            changed_owner_units.update(
                current[unit_id]
                for unit_id in set(current).difference(initial)
            )
        undeclared_changes = changed_owner_units.difference(
            prepared.allowed_blocker_source_unit_ids
        )
        if undeclared_changes:
            raise ProjectedSupportInvariantError(
                f"prepared lifecycle Support topology changed outside declared blockers: {sorted(undeclared_changes)}"
            )

        incumbent_ids = tuple(sorted(inputs.incumbents))
        current_incumbents = {
            memory_id: current_memories[memory_id]
            for memory_id in incumbent_ids
        }
        current_corroboration_targets = {
            claim_hash: current_memories[target.id]
            for claim_hash, target in inputs.corroboration_targets_by_claim_hash.items()
        }
        support_states = await self.db.get_active_memory_support_states(
            memory_ids
        )
        all_support = {
            memory_id: support_states[memory_id].unit_ids
            for memory_id in memory_ids
        }
        support_hashes = {
            memory_id: support_states[memory_id].support_set_hash
            for memory_id in memory_ids
        }
        source_support = await self.db.get_source_unit_support_unit_ids(
            inputs.scope.source_unit_id
        )

        return build_lifecycle_plan(
            plan_id=inputs.plan_id,
            scope=inputs.scope,
            gate_state=inputs.gate_state,
            operations=inputs.operations,
            incumbents=current_incumbents,
            source_support_unit_ids=source_support,
            all_active_support_unit_ids=all_support,
            support_set_hashes=support_hashes,
            observation_revision_ids=inputs.observation_revision_ids,
            evidence_unit_ids_by_claim_hash=(inputs.evidence_unit_ids_by_claim_hash),
            corroboration_targets_by_claim_hash=current_corroboration_targets,
            corroboration_proofs_by_claim_hash=(
                inputs.corroboration_proofs_by_claim_hash
            ),
            defaults=inputs.defaults,
            evidence_units=inputs.evidence_units,
            evidence_references=inputs.evidence_references,
            coordinator_reviews=await self._coordinator_reviews(
                source_unit_id=inputs.scope.source_unit_id, incumbent_ids=incumbent_ids,
            ),
        )

    def _prepared_runtime_bundle(
        self,
        prepared: _PreparedProjectedLifecycleCommit,
        plan: LifecyclePlan,
        *,
        lifecycle_attempt_count: int,
        outcome: str,
        reason_code: str,
    ) -> AgentRuntimeBundle | None:
        if prepared.lifecycle_execution_owner_id is None:
            return None
        scope = prepared.plan_inputs.scope
        return bind_source_lifecycle_outcome(
            source_id=prepared.projection.source_id,
            source_type=prepared.source_type,
            doc_id=prepared.doc_id,
            source_unit_id=scope.source_unit_id,
            base_unit_revision_id=scope.base_unit_revision_id,
            target_unit_revision_id=scope.target_unit_revision_id,
            projection_run_id=prepared.projection.run_id,
            operation_input_hash=prepared.operation_input_hash,
            execution_owner_id=prepared.lifecycle_execution_owner_id,
            outcome=outcome,
            reason_code=reason_code,
            attempt_count=lifecycle_attempt_count,
            duration_ms=max(
                0,
                round((perf_counter() - prepared.started_at) * 1000),
            ),
            incumbent_count=prepared.incumbent_count,
            relation_pair_count=prepared.relation_pair_count,
            mutation_count=len(plan.mutations),
            review_count=sum(
                mutation.mutation_type.value == "create_review"
                for mutation in plan.mutations
            ),
            model_call_count=prepared.model_call_count,
            deployment_revision=current_deployment_revision(),
        )

    async def _commit_prepared_projected_lifecycle(
        self,
        prepared: _PreparedProjectedLifecycleCommit,
        *,
        lifecycle_attempt_count: int,
    ) -> dict[str, int]:
        """Rematerialize stale guards and atomically commit without semantic replay."""

        if lifecycle_attempt_count < 1:
            raise ValueError("lifecycle_attempt_count must be positive")
        if prepared.applied_stats is not None:
            return dict(prepared.applied_stats)
        plan = await self._materialize_prepared_projected_plan(prepared)
        runtime_bundle = self._prepared_runtime_bundle(
            prepared,
            plan,
            lifecycle_attempt_count=lifecycle_attempt_count,
            outcome="expected",
            reason_code="lifecycle_plan_applied",
        )
        try:
            await self.db.apply_source_projection_lifecycle(
                prepared.projection,
                plan,
                document=prepared.document,
                derivation_id=prepared.derivation_id,
                derivation_context_identity_hash=(
                    prepared.derivation_context_identity_hash
                ),
                required_derivation_work_ids=prepared.required_derivation_work_ids,
                expected_source_activity_epoch=(
                    prepared.expected_source_activity_epoch
                ),
                source_activity=prepared.source_activity,
                runtime_bundle=runtime_bundle,
            )
        except ProjectedLifecycleDeferredError as exc:
            if prepared.lifecycle_execution_owner_id is None:
                raise
            failure_bundle = self._prepared_runtime_bundle(
                prepared,
                plan,
                lifecycle_attempt_count=lifecycle_attempt_count,
                outcome="failed",
                reason_code="lifecycle_commit_deferred",
            )
            assert failure_bundle is not None
            raise SourceUnitLifecycleDeferred(
                str(exc),
                failure_bundle,
                handle=DeferredProjectedLifecycleHandle(
                    source_unit_id=prepared.source_unit_id,
                    blocking_source_unit_ids=exc.blocking_source_unit_ids,
                    _prepared=prepared,
                    _runtime_bundle=failure_bundle,
                ),
            ) from exc
        except Exception as exc:
            if prepared.lifecycle_execution_owner_id is None:
                raise
            failure_bundle = self._prepared_runtime_bundle(
                prepared,
                plan,
                lifecycle_attempt_count=lifecycle_attempt_count,
                outcome="failed",
                reason_code=(
                    "authority_plan_stale"
                    if isinstance(exc, AuthorityPlanStaleError)
                    else "lifecycle_commit_failed"
                ),
            )
            assert failure_bundle is not None
            raise SourceUnitLifecycleExecutionError(
                str(exc),
                failure_bundle,
                retryable=not isinstance(exc, ProjectedSupportInvariantError),
            ) from exc

        if runtime_bundle is not None:
            publish_runtime_events(
                self.runtime_event_trace_sink,
                runtime_bundle.events,
            )
            publish_agent_assessments(
                self.agent_assessment_sink,
                runtime_bundle.assessments,
                runtime_bundle.events,
            )
        await self._record_admission_rejections(prepared)
        delivery = await self.memory_store.attempt_lifecycle_vector_delivery(
            plan.id
        )
        stats = dict(prepared.base_stats)
        stats["vector_delivery_pending"] = int(delivery.pending)
        stats["relation_discovery_enqueued"] = len(
            plan.relation_discovery_requests
        )
        for mutation in plan.mutations:
            if mutation.mutation_type.value == "create_memory":
                stats["added"] += 1
            elif mutation.mutation_type.value == "supersede_memory":
                if mutation.payload.get("replacement_kind") == "revision":
                    stats["updated"] += 1
                else:
                    stats["superseded"] += 1
            elif mutation.mutation_type.value == "retire_memory":
                stats["deleted"] += 1
            elif mutation.mutation_type.value == "create_review":
                stats["pending_review"] += 1
        stats["corroborated"] = len(
            {
                mutation.memory_id
                for mutation in plan.mutations
                if mutation.mutation_type.value == "attach_support"
                and mutation.memory_id in prepared.corroboration_target_ids
            }
        )
        stats["noop"] = sum(
            decision.disposition.value == "keep"
            for decision in plan.coverage_proof.incumbent_decisions
        )
        prepared.applied_stats = dict(stats)
        return stats

    async def retry_deferred_projected_lifecycle(
        self,
        handle: DeferredProjectedLifecycleHandle,
        *,
        eligible_same_run_source_unit_ids: set[str] | frozenset[str],
    ) -> dict[str, int]:
        """Authorize and retry one opaque deferred handle without semantic replay."""

        prepared = handle._prepared
        if prepared.applied_stats is not None:
            return dict(prepared.applied_stats)
        eligible = set(eligible_same_run_source_unit_ids)
        if not set(handle.blocking_source_unit_ids).issubset(eligible):
            raise SourceUnitLifecycleExecutionError(
                "deferred lifecycle blocker is outside the current Source run",
                handle._runtime_bundle,
                retryable=False,
                commit_attempted=False,
            )
        prepared.allowed_blocker_source_unit_ids.update(
            handle.blocking_source_unit_ids
        )
        prepared.retry_attempt_count += 1
        return await self._commit_prepared_projected_lifecycle(
            prepared,
            lifecycle_attempt_count=(
                prepared.prepared_at_attempt_count
                + prepared.retry_attempt_count
            ),
        )

    async def _active_support_owners(
        self,
        memory_ids: Sequence[str],
    ) -> dict[str, dict[str, str]]:
        """Return exact active Support ownership for prepared drift guards."""

        owners: dict[str, dict[str, str]] = {
            memory_id: {} for memory_id in memory_ids
        }
        for memory_id in memory_ids:
            for unit in await self.db.get_memory_evidence_units(memory_id):
                owners[memory_id][unit.evidence_unit_id] = unit.source_unit_id
        return owners

    async def _prepare_and_commit_projected_lifecycle_once(
        self,
        *,
        projection: SourceProjection,
        doc_id: str,
        raw_memories: list[RawMemory],
        doc_type: str,
        project_key: str | None,
        repo_identifier: str | None,
        document_content: str,
        update_mode: str,
        changed_hunks: str | None,
        update_plan_stats: dict[str, Any] | None,
        source_updated_at: datetime | None,
        user_id: str | None = None,
        protected_source_observation_ids: tuple[str, ...] = (),
        document: DocumentRecord | None = None,
        derivation_id: str | None = None,
        derivation_reprocess_all_current_observations: bool = False,
        derivation_reprocess_operation_id: str | None = None,
        derivation_support_without_baseline: bool = False,
        expected_source_activity_epoch: int | None = None,
        source_activity: SourceActivityLease | None = None,
        current_changed_ranges: tuple[tuple[int, int], ...] = (),
        lifecycle_execution_owner_id: str | None = None,
        lifecycle_attempt_count: int = 1,
        _runtime_context: _LifecycleExecutionContext,
    ) -> dict[str, int]:
        """Reconcile a complete Source Unit ledger and atomically apply one plan."""

        from memforge.pipeline.projection_evidence import build_projected_claim_evidence

        lifecycle_started = _runtime_context.started_at
        if lifecycle_attempt_count < 1:
            raise ValueError("lifecycle_attempt_count must be positive")
        if len(projection.deltas) != 1:
            raise ValueError("projected lifecycle requires exactly one Revision Delta")
        delta = projection.deltas[0]
        scope = ReconciliationScope(
            id=f"scope:{projection.run_id}",
            source_id=projection.source_id,
            source_unit_id=delta.source_unit_id,
            base_unit_revision_id=delta.previous_unit_revision_id,
            target_unit_revision_id=delta.current_unit_revision_id,
        )
        observation_revision_ids = tuple(revision.id for revision in projection.observation_revisions)
        source_type = projection.source_type

        stats = {
            "added": 0,
            "corroborated": 0,
            "updated": 0,
            "superseded": 0,
            "deleted": 0,
            "noop": 0,
            "pending_review": 0,
            "skipped": 0,
            "vector_delivery_pending": 0,
            "relation_discovery_enqueued": 0,
            "support_revalidation_work_item_count": 0,
            "support_revalidation_model_call_count": 0,
            "support_revalidation_revision_index_count": 0,
            "support_revalidation_prompt_chars": 0,
            "support_revalidation_supported_count": 0,
            "support_revalidation_program_rebind_count": 0,
            "support_revalidation_change_impact_request_count": 0,
            "support_revalidation_change_impact_unaffected_count": 0,
            "support_revalidation_change_impact_affected_count": 0,
            "support_revalidation_change_impact_failed_count": 0,
            "support_revalidation_unresolved_partial_coverage_count": 0,
            "support_revalidation_unresolved_capacity_count": 0,
            "support_revalidation_unusable_baseline_count": 0,
            "support_revalidation_reprocess_count": 0,
        }
        filtered_memories: list[RawMemory] = []
        for raw in raw_memories:
            if self._candidate_can_persist(
                raw,
                stats,
                observation_semantic_class=_observation_semantic_class(
                    projection,
                    raw.source_observation_id,
                ),
            ):
                filtered_memories.append(raw)
        quality_candidate_count = len(filtered_memories)
        incumbents, unit_support = await self._active_projected_incumbents(
            doc_id=doc_id,
            source_unit_id=scope.source_unit_id,
        )
        gate = await self.db.get_lifecycle_gate(scope.source_id)
        incumbent_support_states = await self.db.get_active_memory_support_states(
            tuple(memory.id for memory in incumbents)
        )
        support_hashes = {
            memory_id: state.support_set_hash
            for memory_id, state in incumbent_support_states.items()
        }
        operation_input_hash = _source_lifecycle_operation_input_hash(
            projection=projection,
            candidates=filtered_memories,
            incumbents=incumbents,
            support_hashes=support_hashes,
            gate_state=gate.state.value,
            update_mode=update_mode,
            changed_hunks=changed_hunks,
            update_plan_stats=update_plan_stats,
            llm_model=self.llm_model,
            input_policy_identity=getattr(self.structured_llm_client, "input_policy_identity", None),
        )
        _runtime_context.operation_input_hash = operation_input_hash
        enrich_failure_trace_context(operation_input_hash=operation_input_hash)
        if lifecycle_execution_owner_id is not None:
            enrich_failure_trace_context(**source_lifecycle_execution_identity(
                source_id=projection.source_id, source_unit_id=scope.source_unit_id,
                base_unit_revision_id=scope.base_unit_revision_id,
                target_unit_revision_id=scope.target_unit_revision_id,
                operation_input_hash=operation_input_hash, execution_owner_id=lifecycle_execution_owner_id))
        _runtime_context.incumbent_count = len(incumbents)
        _runtime_context.stage = "candidate_admission"
        evidence_image_loader = _projection_evidence_image_loader(projection, self.document_store)
        admission = await admit_candidates(
            filtered_memories,
            client=self.structured_llm_client,
            model=self.llm_model,
            image_loader=evidence_image_loader,
            store=self.db,
            derivation_id=derivation_id,
            operation_input_hash=operation_input_hash,
        )
        filtered_memories = list(admission.admitted)
        stats.update(
            {
                "candidate_admission_admitted_count": len(admission.admitted),
                "candidate_admission_rejected_count": len(admission.rejected),
                "candidate_admission_merged_count": admission.merged_count,
                "candidate_admission_llm_calls": admission.llm_calls,
                "candidate_admission_prompt_chars": admission.prompt_chars,
            }
        )
        stats["skipped"] += quality_candidate_count - len(filtered_memories)
        _runtime_context.model_call_count += admission.llm_calls
        _runtime_context.stage = "reconciliation"
        reconciliation_started = perf_counter()
        derivation_protected_ids = await self._derivation_protected_incumbents(
            source_id=projection.source_id,
            incumbent_ids=frozenset(memory.id for memory in incumbents),
            protected_source_observation_ids=frozenset(protected_source_observation_ids),
        )
        incumbent_impacts: dict[str, ImpactResult] = {}
        if projection.coverage is ProjectionCoverage.PARTIAL_PROJECTION:
            incumbent_impacts = await self._projected_incumbent_impacts(
                projection=projection,
                incumbent_ids=frozenset(memory.id for memory in incumbents),
                unit_support=unit_support,
            )
        visibility, owner_user_id = await memory_visibility_for_source_id(self.db, source_id=projection.source_id)
        if visibility == "private" and user_id is not None and user_id != owner_user_id:
            raise PermissionError("private projected lifecycle actor does not own the document")
        access_context_hash = lifecycle_access_context_hash(
            visibility=visibility, owner_user_id=owner_user_id,
            project_key=project_key, repo_identifier=repo_identifier,
        )
        supports: dict[str, MemorySupport] = {}
        required_derivation_work_ids = admission.work_ids
        model_incumbent_count = 0
        model_batch_count = 0
        structured_llm_call_count = 0
        structured_llm_elapsed_ms = 0
        bounded_reconciliation_elapsed_ms = 0
        from memforge.pipeline.projection_images import projection_inference_image_observation_ids
        if not document_content.strip() and not filtered_memories and not projection_inference_image_observation_ids(projection):
            operations = tuple(
                ReconcileOperation(
                    action=ReconcileAction.DELETE,
                    memory_id=memory.id,
                    reason=(
                        "current Source Artifact revision is not inference eligible"
                        if memory.id in derivation_protected_ids
                        else "source observation is explicitly empty"
                    ),
                    flag_for_review=(memory.id in derivation_protected_ids),
                )
                for memory in sorted(incumbents, key=lambda item: item.id)
            )
        else:
            from memforge.pipeline.reconciler import (
                ReconciliationContractError,
                ReconciliationResult,
                assess_relations,
                join_support_and_relation,
            )
            from memforge.pipeline.revision_work import RevisionWorkExecutor, SupportRecheck
            from memforge.pipeline.support_reading import SupportWorkItem, evidence_digests

            model_incumbents = [memory for memory in incumbents if memory.id not in derivation_protected_ids]
            model_incumbent_count = len(model_incumbents)
            if model_incumbents and not self.structured_llm_client:
                raise RuntimeError("complete lifecycle reconciliation requires an LLM client")
            work_items: list[SupportWorkItem] = []
            work_by_memory: dict[str, list[SupportWorkItem]] = {}
            evaluator = RevisionWorkExecutor(client=self.structured_llm_client, model=self.llm_model,
                                             store=self.db, derivation_id=derivation_id)
            assessment_context = None
            if model_incumbents:
                base = await self.db.get_current_source_unit_projection(scope.source_unit_id)
                if base is not None and base.source_unit_revisions[0].id not in {scope.base_unit_revision_id, scope.target_unit_revision_id}:
                    raise AuthorityPlanStaleError("revision assessment base changed")
                assessment_context = RevisionAssessmentContext(
                    projection=projection, base=base, access_context_hash=access_context_hash, image_loader=evidence_image_loader,
                )

                def assessment_context_for(baseline):
                    """Assess against another baseline, sharing this operation's Fragment indexes.

                    The committed revision still describes carried Observations the baseline may predate.
                    """
                    return RevisionAssessmentContext(
                        projection=projection, base=baseline, access_context_hash=access_context_hash,
                        image_loader=evidence_image_loader, indexes=assessment_context.indexes,
                        known_observations=base.observations if base is not None else (),
                    )
                contexts_by_revision = {base.source_unit_revisions[0].id: assessment_context} if base else {}
                evidence_by_memory = await self.db.get_active_memory_support_evidence_many(
                    tuple(memory.id for memory in model_incumbents), source_id=projection.source_id,
                )
                for memory in model_incumbents:
                    groups: dict[str, list] = {}
                    for item in evidence_by_memory.get(memory.id, ()):
                        if item.evidence_unit_id in unit_support.get(memory.id, ()):
                            groups.setdefault(item.evidence_unit_id, []).append(item)
                    if not groups:
                        raise ReconciliationContractError("revision_support_missing", "incumbent has no complete scoped support")
                    work_by_memory[memory.id] = []
                    for evidence_unit_id, support in groups.items():
                        if derivation_support_without_baseline:
                            # An operator reprocess reads the whole Unit for every Support.
                            baseline_id, unusable = None, None
                            stats["support_revalidation_reprocess_count"] += 1
                        else:
                            baseline_id, unusable = _support_validation_baseline(support)
                        context = contexts_by_revision.get(baseline_id) if unusable is None else None
                        if context is None and baseline_id is not None:
                            historical = await self.db.get_source_unit_revision_projection(
                                scope.source_unit_id, baseline_id
                            )
                            if (
                                historical is None
                                or historical.source_id != projection.source_id
                                or len(historical.source_unit_revisions) != 1
                                or historical.source_unit_revisions[0].source_unit_id != scope.source_unit_id
                                or historical.source_unit_revisions[0].id != baseline_id
                            ):
                                unusable = "validation snapshot is unavailable or inconsistent"
                            else:
                                context = assessment_context_for(historical)
                                contexts_by_revision[baseline_id] = context
                        if context is None:
                            # No usable baseline: the whole current revision is read without
                            # assuming the old Support is still valid. A named but unusable
                            # baseline is a data defect worth a diagnostic, not a failure.
                            if unusable is not None:
                                logger.warning(
                                    "support_baseline_unusable memory_id=%s evidence_unit_id=%s "
                                    "baseline_unit_revision_ids=%s reason=%s",
                                    memory.id, evidence_unit_id,
                                    ",".join(sorted({str(item.validation_unit_revision_id) for item in support})),
                                    unusable,
                                )
                                stats["support_revalidation_unusable_baseline_count"] += 1
                            context = contexts_by_revision.get(None)
                            if context is None:
                                context = assessment_context_for(None)
                                contexts_by_revision[None] = context
                        work_item = SupportWorkItem(f"w{len(work_items):06d}", memory, tuple(support), context)
                        work_items.append(work_item)
                        work_by_memory[memory.id].append(work_item)
            stats["support_revalidation_work_item_count"] += len(work_items)
            assessed: dict[str, SupportAssessment] = {}

            async def recheck(requests: tuple[SupportRecheckRequest, ...]) -> dict[str, MemorySupport]:
                """Run each conflicting claim's single re-check and fold it into its Memory-level result."""
                rechecks: list[SupportRecheck] = []
                owners: dict[str, list[str]] = {}
                for request in requests:
                    memory_items = work_by_memory[request.memory_id]
                    if request.reading is RecheckReading.NORMAL_ORDER:
                        rebound = [item for item in memory_items if assessed[item.id].rebound]
                        rechecks.extend(SupportRecheck(item) for item in rebound)
                        owners[request.memory_id] = [item.id for item in rebound]
                        continue
                    # One read of the claim, carrying every prior part it has in this Unit.
                    item = SupportWorkItem(
                        f"r{len(rechecks):06d}", memory_items[0].memory,
                        tuple(part for memory_item in memory_items for part in memory_item.support),
                        memory_items[0].context,
                    )
                    rechecks.append(SupportRecheck(item, tuple(sorted({
                        digest for raw in request.candidates for digest in evidence_digests(raw.resolved_evidence_selection)
                    }))))
                    owners[request.memory_id] = [item.id]
                results = await evaluator.recheck(rechecks)
                rechecked = {}
                for request in requests:
                    assessments = [results[work_id] for work_id in owners[request.memory_id]]
                    if request.reading is RecheckReading.NORMAL_ORDER:
                        read = [assessed[item.id] for item in work_by_memory[request.memory_id] if not assessed[item.id].rebound]
                        rechecked[request.memory_id] = memory_support([*read, *assessments], rechecked=True)
                    else:
                        rechecked[request.memory_id] = supports[request.memory_id].after_candidate_recheck(assessments[0])
                supports.update(rechecked)
                return rechecked

            async def support_line() -> None:
                if work_items:
                    assessed.update(await evaluator.assess_many(work_items))

            relation_line = assess_relations(
                filtered_memories, model_incumbents, structured_llm_client=self.structured_llm_client,
                llm_model=self.llm_model, image_loader=evidence_image_loader, work_store=self.db,
                derivation_id=derivation_id, operation_input_hash=operation_input_hash,
            )
            # A failure raised on either line is a Support failure: Relation returns its own.
            _runtime_context.stage = "support_revalidation"
            try:
                _, relation = await _run_concurrently(support_line(), relation_line)
                unresolved_stats = {
                    "partial_coverage": "support_revalidation_unresolved_partial_coverage_count",
                    "capacity": "support_revalidation_unresolved_capacity_count",
                }
                for assessment in assessed.values():
                    if assessment.unresolved is not None:
                        stats[unresolved_stats[assessment.unresolved]] += 1
                for memory in model_incumbents:
                    supports[memory.id] = memory_support([assessed[item.id] for item in work_by_memory[memory.id]])
                    stats["support_revalidation_supported_count"] += len(supports[memory.id].evidence)
                if assessment_context is not None:
                    stats["support_revalidation_revision_index_count"] = len(assessment_context.indexes)
                if relation.failure is not None:
                    result = ReconciliationResult(operations=[], failure=relation.failure, metrics=relation.metrics)
                else:
                    ledger = carry_into_ledger(
                        filtered_memories, relation.entries,
                        await self._carried_conflicts(
                            source_unit_id=scope.source_unit_id, model_incumbents=model_incumbents,
                            context=assessment_context,
                        ),
                    )
                    stats["coordinator_carried_conflict_count"] = len(ledger.carried_pairs)
                    result = await join_support_and_relation(
                        replace(relation, entries=ledger.entries),
                        new_extractions=ledger.candidates,
                        existing_memories=model_incumbents,
                        supports=dict(supports),
                        structured_llm_client=self.structured_llm_client,
                        llm_model=self.llm_model,
                        recheck=recheck,
                        rechecked_pairs=ledger.carried_pairs,
                    )
            finally:
                stats["support_revalidation_model_call_count"] += evaluator.calls
                stats["support_revalidation_assessment_request_count"] = evaluator.stage_counts["support_assess"]
                stats["support_revalidation_prompt_chars"] += evaluator.prompt_chars
                stats["support_revalidation_reused_work_count"] = evaluator.reused
                stats["support_revalidation_covered_source_claim_pairs"] = evaluator.covered_source_claim_pairs
                stats["support_revalidation_program_rebind_count"] = evaluator.program_rebind_count
                stats["support_revalidation_change_impact_request_count"] = evaluator.stage_counts["change_impact"]
                impact_counts = evaluator.change_impact_counts
                stats["support_revalidation_change_impact_unaffected_count"] = impact_counts["unaffected"]
                stats["support_revalidation_change_impact_affected_count"] = impact_counts["affected"]
                stats["support_revalidation_change_impact_failed_count"] = impact_counts["failed"]
                _runtime_context.model_call_count += evaluator.calls
            stats["support_revalidation_completion_count"] = len(evaluator.final_work_ids)
            if derivation_id:
                required_derivation_work_ids += tuple(evaluator.final_work_ids)
            _runtime_context.stage = "reconciliation"
            reconciliation_metrics = result.metrics
            required_derivation_work_ids += result.work_ids
            model_batch_count = reconciliation_metrics.model_batch_count
            structured_llm_call_count = reconciliation_metrics.structured_llm_calls
            structured_llm_elapsed_ms = reconciliation_metrics.structured_llm_elapsed_ms
            bounded_reconciliation_elapsed_ms = reconciliation_metrics.reconciliation_elapsed_ms
            _runtime_context.relation_pair_count = reconciliation_metrics.relation_pair_count
            _runtime_context.model_call_count += reconciliation_metrics.structured_llm_calls
            stats.update(
                {
                    "reconciliation_relation_pair_count": reconciliation_metrics.relation_pair_count,
                    "reconciliation_relation_prompt_chars": reconciliation_metrics.relation_prompt_chars,
                    "reconciliation_revision_proof_count": reconciliation_metrics.revision_proof_count,
                    "reconciliation_revision_proof_failure_count": (
                        reconciliation_metrics.revision_proof_failure_count
                    ),
                    "coordinator_recheck_count": len(result.rechecks),
                    "coordinator_review_count": sum(len(operation.reviews) for operation in result.operations),
                    "coordinator_unresolved_candidate_count": result.unresolved_candidate_count,
                }
            )
            if result.failure is not None:
                message = (
                    f"complete lifecycle reconciliation failed: {result.failure.error_type}: {result.failure.error}"
                )
                if lifecycle_execution_owner_id is None:
                    raise RuntimeError(message)
                failure_bundle = bind_source_lifecycle_outcome(
                    source_id=projection.source_id,
                    source_type=source_type,
                    doc_id=doc_id,
                    source_unit_id=scope.source_unit_id,
                    base_unit_revision_id=scope.base_unit_revision_id,
                    target_unit_revision_id=scope.target_unit_revision_id,
                    projection_run_id=projection.run_id,
                    operation_input_hash=operation_input_hash,
                    execution_owner_id=lifecycle_execution_owner_id,
                    outcome="failed",
                    reason_code=result.failure.reason_code,
                    operation=result.failure.operation,
                    terminal_category=result.failure.terminal_category,
                    error_code=result.failure.error_code,
                    validation_fields=result.failure.validation_fields,
                    diagnostic=result.failure.diagnostic,
                    attempt_count=lifecycle_attempt_count,
                    duration_ms=max(0, round((perf_counter() - lifecycle_started) * 1000)),
                    incumbent_count=len(incumbents),
                    relation_pair_count=result.metrics.relation_pair_count,
                    mutation_count=0,
                    review_count=0,
                    model_call_count=_runtime_context.model_call_count,
                    deployment_revision=current_deployment_revision(),
                )
                raise SourceUnitLifecycleExecutionError(
                    message, failure_bundle,
                    retryable=result.failure.terminal_category in {"provider_error", "deadline_exceeded"},
                    commit_attempted=False,
                )
            operations = tuple(result.operations) + tuple(
                ReconcileOperation(
                    action=ReconcileAction.DELETE,
                    memory_id=memory_id,
                    reason=("current Source Artifact revision is not inference eligible"),
                    flag_for_review=True,
                )
                for memory_id in sorted(derivation_protected_ids)
            )
        stats["support_revalidation_skipped_memory_count"] = sum(
            operation.support_revalidation_skipped for operation in operations
        )
        _runtime_context.stage = "plan_construction"
        protected_memory_ids = self._partial_projection_protected_incumbents(
            projection=projection,
            incumbent_impacts=incumbent_impacts,
        )
        operations = self._enforce_partial_projection_keep(
            operations,
            protected_memory_ids,
        )
        logger.info(
            json.dumps(
                {
                    "event": "projected_lifecycle_reconciliation",
                    "source_id": projection.source_id,
                    "source_unit_id": scope.source_unit_id,
                    "reconciliation_new_candidate_count": len(filtered_memories),
                    "reconciliation_incumbent_count": len(incumbents),
                    "reconciliation_model_incumbent_count": model_incumbent_count,
                    "reconciliation_llm_batch_count": model_batch_count,
                    "reconciliation_llm_call_count": structured_llm_call_count,
                    "reconciliation_llm_elapsed_ms": structured_llm_elapsed_ms,
                    "reconciliation_bounded_elapsed_ms": (bounded_reconciliation_elapsed_ms),
                    "reconciliation_relation_pair_count": stats.get(
                        "reconciliation_relation_pair_count", 0
                    ),
                    "reconciliation_relation_prompt_chars": stats.get(
                        "reconciliation_relation_prompt_chars", 0
                    ),
                    "reconciliation_revision_proof_count": stats.get(
                        "reconciliation_revision_proof_count", 0
                    ),
                    "reconciliation_revision_proof_failure_count": stats.get(
                        "reconciliation_revision_proof_failure_count", 0
                    ),
                    "reconciliation_total_elapsed_ms": max(
                        0,
                        round((perf_counter() - reconciliation_started) * 1000),
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        proposed_memories = [
            *(
                operation.memory for operation in operations
                if operation.action in {ReconcileAction.ADD, ReconcileAction.UPDATE, ReconcileAction.SUPERSEDE}
                and operation.memory is not None
            ),
            *(
                review.candidate for operation in operations for review in operation.reviews
                if review.proposal is CoordinatorProposal.SUPERSEDE
            ),
        ]
        for proposed in proposed_memories:
            quality = classify_memory_candidate(proposed)
            if not quality.keep:
                raise RuntimeError(
                    "complete lifecycle reconciliation produced an unsafe Memory candidate: "
                    f"{quality.skip_reason or 'quality_rejected'}"
                )
        incumbents_by_id = {memory.id: memory for memory in incumbents}
        _runtime_context.stage = "plan_construction"
        corroboration_targets: dict[str, Memory] = {}
        corroboration_proofs: dict[str, dict[str, object]] = {}
        identity_claim_hashes: list[str] = []
        identity_excluded_ids = frozenset(incumbents_by_id).difference(
            operation.memory_id for operation in operations
            if operation.support_revalidation_skipped and not operation.reviews
        )
        identity_requests: list[IdentityResolutionRequest] = []
        operation_memories = tuple(operation.memory for operation in operations if operation.memory is not None)
        memory_texts_by_mention: dict[str, list[str]] = {}
        for raw_memory in operation_memories:
            for entity_ref in raw_memory.entity_refs:
                memory_texts_by_mention.setdefault(entity_ref, []).append(raw_memory.content)
        entity_resolution = await self.entity_resolver.resolve_many(
            memory_texts_by_mention,
            scope=EntityResolutionScope(access_context_hash=access_context_hash),
        )
        stats.update(
            {
                "entity_resolution_unique_mentions": entity_resolution.metrics.unique_mentions,
                "entity_resolution_exact_hits": entity_resolution.metrics.exact_hits,
                "entity_resolution_alias_hits": entity_resolution.metrics.alias_hits,
                "entity_resolution_embedded_mentions": entity_resolution.metrics.embedded_mentions,
                "entity_resolution_ambiguous_mentions": entity_resolution.metrics.ambiguous_mentions,
                "entity_resolution_embedding_batches": entity_resolution.metrics.embedding_batches,
                "entity_resolution_llm_calls": entity_resolution.metrics.structured_llm_calls,
                "entity_resolution_validation_retries": (
                    entity_resolution.metrics.validation_retries
                ),
                "entity_resolution_candidate_count": entity_resolution.metrics.candidate_count,
                "entity_resolution_new_entities": entity_resolution.metrics.new_entities,
                "entity_resolution_elapsed_ms": entity_resolution.metrics.elapsed_ms,
            }
        )
        entity_ids_by_claim_hash = {
            content_hash(raw_memory.content.strip()): tuple(
                dict.fromkeys(
                    entity_id
                    for entity_ref in raw_memory.entity_refs
                    if (entity_id := entity_resolution.entity_id(entity_ref)) is not None
                )
            )
            for raw_memory in operation_memories
        }
        for operation in operations:
            if operation.action is not ReconcileAction.ADD or operation.memory is None:
                continue
            candidate = self._build_memory(
                operation.memory,
                project_key,
                visibility=visibility,
                owner_user_id=owner_user_id,
                repo_identifier=repo_identifier,
            )
            identity_claim_hashes.append(content_hash(operation.memory.content.strip()))
            identity_requests.append(
                IdentityResolutionRequest(
                    challenger=candidate,
                    doc_id=doc_id,
                    entity_ids=entity_ids_by_claim_hash.get(
                        content_hash(operation.memory.content.strip()),
                        (),
                    ),
                    excluded_memory_ids=identity_excluded_ids,
                )
            )
        identity_resolution = await self.identity_resolver.resolve(tuple(identity_requests))
        identity_resolutions = identity_resolution.resolutions
        stats.update(
            {
                "identity_resolution_pair_count": identity_resolution.metrics.pair_count,
                "identity_resolution_llm_calls": identity_resolution.metrics.llm_calls,
                "identity_resolution_prompt_chars": identity_resolution.metrics.prompt_chars,
                "identity_resolution_elapsed_ms": identity_resolution.metrics.elapsed_ms,
            }
        )
        incomplete_identity = next((item for item in identity_resolutions if not item.classification_complete), None)
        if incomplete_identity is not None:
            raise MemoryPairClassificationError(
                incomplete_identity.failure_reason or "identity discovery did not complete",
                pair_count=identity_resolution.metrics.pair_count,
                llm_calls=identity_resolution.metrics.llm_calls,
                prompt_chars=identity_resolution.metrics.prompt_chars,
                terminal_category=incomplete_identity.terminal_category,
                error_code=incomplete_identity.error_code,
            )
        attached_target_ids: list[str] = []
        for claim_hash, resolution in zip(
            identity_claim_hashes,
            identity_resolutions,
            strict=True,
        ):
            target = resolution.target
            equivalence_proof = resolution.equivalence_proof
            if target is None or equivalence_proof is None:
                continue
            corroboration_targets[claim_hash] = target
            corroboration_proofs[claim_hash] = dict(equivalence_proof)
            attached_target_ids.append(target.id)
        evidence_memories = [operation.memory for operation in operations if operation.memory is not None]
        for operation in operations:
            if operation.action is not ReconcileAction.NOOP:
                continue
            support = supports.get(operation.memory_id or "")
            if operation.memory is not None and support is not None and support.evidence[:1] == (operation.memory,):
                evidence_memories.extend(support.evidence[1:])
            for review in operation.reviews:
                evidence_memories.append(review.candidate)
                if review.rejection_rebind is not None:
                    evidence_memories.append(review.rejection_rebind)
        projected_evidence = build_projected_claim_evidence(
            projection=projection,
            raw_memories=evidence_memories,
            doc_id=doc_id,
            source_type=source_type,
            project_key=project_key,
            visibility=visibility,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            access_context_hash=access_context_hash,
            extractor_run_id=projection.run_id,
            observed_at=(source_updated_at.isoformat() if source_updated_at is not None else None),
        )
        def canonical(raw: RawMemory) -> RawMemory:
            return projected_evidence.canonical_memories_by_claim_hash[content_hash(raw.content.strip())]

        operations = tuple(
            replace(
                operation,
                memory=canonical(operation.memory) if operation.memory is not None else None,
                reviews=tuple(
                    replace(
                        review,
                        candidate=canonical(review.candidate),
                        rejection_rebind=(
                            canonical(review.rejection_rebind) if review.rejection_rebind is not None else None
                        ),
                    )
                    for review in operation.reviews
                ),
            )
            for operation in operations
        )
        defaults = NewMemoryDefaults(
            visibility=visibility,
            owner_user_id=owner_user_id,
            project_key=project_key,
            repo_identifier=repo_identifier,
            doc_id=doc_id,
            source_type=source_type,
            access_context_hash=access_context_hash,
            actor_user_id=user_id,
            entity_ids_by_claim_hash=entity_ids_by_claim_hash,
            source_updated_at=(
                source_updated_at.isoformat()
                if source_updated_at is not None
                else None
            ),
        )
        prepared_memories = {
            **incumbents_by_id,
            **{
                target.id: target
                for target in corroboration_targets.values()
            },
        }
        initial_support_owners = await self._active_support_owners(
            tuple(sorted(prepared_memories))
        )
        derivation_context_identity_hash = (
            source_derivation_context_identity_hash(
                SourceUnitDerivationContext(
                    document=document,
                    doc_type=doc_type,
                    project_key=project_key,
                    repo_identifier=repo_identifier,
                    document_content=document_content,
                    update_mode=update_mode,
                    changed_hunks=changed_hunks,
                    update_plan_stats=update_plan_stats,
                    source_updated_at=(
                        source_updated_at.isoformat()
                        if source_updated_at is not None
                        else None
                    ),
                    user_id=user_id,
                    source_activity_epoch=expected_source_activity_epoch,
                    current_changed_ranges=current_changed_ranges,
                    reprocess_all_current_observations=(
                        derivation_reprocess_all_current_observations
                    ),
                    reprocess_operation_id=(
                        derivation_reprocess_operation_id
                    ),
                    support_without_baseline=derivation_support_without_baseline,
                )
            )
            if derivation_id is not None and document is not None
            else None
        )
        prepared = _PreparedProjectedLifecycleCommit(
            projection=projection,
            plan_inputs=_PreparedLifecyclePlanInputs(
                plan_id=lifecycle_plan_id(scope),
                scope=scope,
                gate_state=gate.state,
                operations=operations,
                incumbents=incumbents_by_id,
                memory_authority_hashes={
                    memory_id: _prepared_memory_authority_hash(memory)
                    for memory_id, memory in prepared_memories.items()
                },
                initial_support_owners=initial_support_owners,
                observation_revision_ids=observation_revision_ids,
                evidence_unit_ids_by_claim_hash=(
                    projected_evidence.evidence_unit_ids_by_claim_hash
                ),
                corroboration_targets_by_claim_hash=corroboration_targets,
                corroboration_proofs_by_claim_hash=corroboration_proofs,
                defaults=defaults,
                evidence_units=projected_evidence.units,
                evidence_references=projected_evidence.references,
            ),
            document=document,
            derivation_id=derivation_id,
            derivation_context_identity_hash=derivation_context_identity_hash,
            required_derivation_work_ids=required_derivation_work_ids,
            expected_source_activity_epoch=expected_source_activity_epoch,
            source_activity=source_activity,
            base_stats=dict(stats),
            corroboration_target_ids=frozenset(attached_target_ids),
            lifecycle_execution_owner_id=lifecycle_execution_owner_id,
            operation_input_hash=operation_input_hash,
            doc_id=doc_id,
            source_type=source_type,
            started_at=lifecycle_started,
            incumbent_count=len(incumbents),
            relation_pair_count=int(
                stats.get("reconciliation_relation_pair_count", 0)
            ),
            model_call_count=(
                admission.llm_calls
                + structured_llm_call_count
                + int(stats["support_revalidation_model_call_count"])
                + entity_resolution.metrics.structured_llm_calls
                + identity_resolution.metrics.llm_calls
            ),
            prepared_at_attempt_count=lifecycle_attempt_count,
            admission_rejections=admission.rejected,
        )
        _runtime_context.stage = "lifecycle_commit"
        return await self._commit_prepared_projected_lifecycle(
            prepared,
            lifecycle_attempt_count=lifecycle_attempt_count,
        )

    async def _record_admission_rejections(self, prepared: _PreparedProjectedLifecycleCommit) -> None:
        """Record one event per Candidate the committed revision rejected."""

        projection = prepared.projection
        revision = projection.source_unit_revisions[0]
        for rejection in prepared.admission_rejections:
            await self.memory_store.record_audit_event(
                "candidate_admission_rejected",
                "committed",
                context=self.memory_store.operation_context(
                    run_id=projection.run_id, source_id=projection.source_id, doc_id=prepared.doc_id,
                ),
                doc_id=prepared.doc_id,
                source_id=projection.source_id,
                decision="reject_candidate",
                reason=rejection.reject_reason,
                payload=_candidate_rejection_payload(rejection, revision),
            )

    async def apply_projected_tombstone(
        self,
        *,
        projection: SourceProjection,
        doc_id: str,
        reason: str,
        lifecycle_cycle_id: str,
        expected_source_activity_epoch: int | None = None,
        source_activity: SourceActivityLease | None = None,
    ) -> dict[str, int | bool]:
        """Apply an authoritative Source Unit tombstone without an LLM call.

        Provider absence is already an explicit deterministic fact at this
        boundary. Every active same-document incumbent therefore receives a
        DELETE ledger entry, while the per-source lifecycle gate still decides
        whether that becomes support removal/retirement or a durable review.
        """

        if len(projection.deltas) != 1 or not projection.coverage.proves_absence:
            raise ValueError("projected tombstone requires one absence-proving Revision Delta")
        if not lifecycle_cycle_id.strip():
            raise ValueError("projected tombstone requires lifecycle cycle identity")
        delta = projection.deltas[0]
        scope = ReconciliationScope(
            id=(f"tombstone:{lifecycle_cycle_id}:{delta.source_unit_id}:{delta.current_unit_revision_id or 'removed'}"),
            source_id=projection.source_id,
            source_unit_id=delta.source_unit_id,
            base_unit_revision_id=delta.previous_unit_revision_id,
            target_unit_revision_id=delta.current_unit_revision_id,
        )
        plan_id = lifecycle_plan_id(scope)
        applied_payload = await self.db.get_lifecycle_plan_payload(plan_id)
        if applied_payload is not None:
            stored_scope = applied_payload.get("scope")
            mutations = applied_payload.get("mutations")
            if (
                not isinstance(stored_scope, Mapping)
                or stored_scope.get("id") != scope.id
                or stored_scope.get("source_id") != scope.source_id
                or stored_scope.get("source_unit_id") != scope.source_unit_id
                or stored_scope.get("target_unit_revision_id") != scope.target_unit_revision_id
                or not isinstance(mutations, list)
            ):
                raise ValueError("applied tombstone lifecycle ledger is malformed")
            mutation_types = [mutation.get("mutation_type") for mutation in mutations if isinstance(mutation, Mapping)]
            if len(mutation_types) != len(mutations):
                raise ValueError("applied tombstone lifecycle mutation ledger is malformed")
            await self.memory_store.attempt_lifecycle_vector_delivery(plan_id)
            return await self._projected_tombstone_result(
                doc_id=doc_id,
                mutation_types=mutation_types,
            )
        source_type = projection.source_type
        incumbents, unit_support = await self._active_projected_incumbents(
            doc_id=doc_id,
            source_unit_id=scope.source_unit_id,
        )
        incumbents_by_id = {memory.id: memory for memory in incumbents}
        operations = tuple(
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=memory_id,
                reason=reason,
            )
            for memory_id in sorted(incumbents_by_id)
        )
        gate = await self.db.get_lifecycle_gate(scope.source_id)
        support_states = await self.db.get_active_memory_support_states(tuple(incumbents_by_id))
        all_support = {memory_id: state.unit_ids for memory_id, state in support_states.items()}
        support_hashes = {memory_id: state.support_set_hash for memory_id, state in support_states.items()}
        visibility, owner_user_id = await memory_visibility_for_document(self.db, doc_id=doc_id)
        plan = build_lifecycle_plan(
            plan_id=plan_id,
            scope=scope,
            gate_state=gate.state,
            operations=operations,
            incumbents=incumbents_by_id,
            source_support_unit_ids=unit_support,
            all_active_support_unit_ids=all_support,
            support_set_hashes=support_hashes,
            observation_revision_ids=(),
            defaults=NewMemoryDefaults(
                visibility=visibility,
                owner_user_id=owner_user_id,
                project_key=None,
                repo_identifier=None,
                doc_id=doc_id,
                source_type=source_type,
                access_context_hash=lifecycle_access_context_hash(
                    visibility=visibility,
                    owner_user_id=owner_user_id,
                    project_key=None,
                    repo_identifier=None,
                ),
            ),
        )
        await self.db.apply_source_projection_lifecycle(
            projection,
            plan,
            expected_source_activity_epoch=expected_source_activity_epoch,
            source_activity=source_activity,
        )
        await self.memory_store.attempt_lifecycle_vector_delivery(plan.id)
        return await self._projected_tombstone_result(
            doc_id=doc_id,
            mutation_types=tuple(mutation.mutation_type.value for mutation in plan.mutations),
        )

    async def _projected_tombstone_result(
        self,
        *,
        doc_id: str,
        mutation_types: Sequence[str],
    ) -> dict[str, int | bool]:
        """Return deletion eligibility from the committed document provenance."""

        pending_review = mutation_types.count("create_review")
        remaining_document_support = await self.db.get_memories_by_source_doc(
            doc_id,
            support_kind=None,
        )
        return {
            "retired": mutation_types.count("retire_memory"),
            "pending_review": pending_review,
            "can_delete_document": (pending_review == 0 and not remaining_document_support),
        }

    def _candidate_can_persist(
        self,
        raw: RawMemory,
        stats: dict | None = None,
        *,
        observation_semantic_class: str | None = None,
    ) -> bool:
        """Return whether a raw candidate should be persisted, updating stats when skipped."""
        quality = classify_memory_candidate(
            raw,
            observation_semantic_class=observation_semantic_class,
        )
        if quality.keep:
            return True

        if stats is not None:
            stats["skipped"] = stats.get("skipped", 0) + 1
        logger.info(
            "Skipping memory candidate (%s): %s",
            quality.skip_reason,
            raw.content.strip()[:120],
        )
        return False

    def _build_memory(
        self,
        raw: RawMemory,
        project_key: str | None,
        *,
        visibility: str,
        owner_user_id: str | None,
        repo_identifier: str | None = None,
        memory_id: str | None = None,
    ) -> Memory:
        """Build a Memory object from a RawMemory."""
        return Memory(
            id=memory_id or generate_memory_id(),
            memory_type=raw.memory_type,
            content=raw.content.strip(),
            content_hash=content_hash(raw.content.strip()),
            visibility=visibility,
            owner_user_id=owner_user_id,
            project_key=project_key,
            repo_identifier=repo_identifier,
            entity_refs=raw.entity_refs,
            confidence=raw.confidence,
            corroboration_count=1,
            valid_from=parse_memory_validity_date(raw.valid_from),
            valid_until=parse_memory_validity_date(raw.valid_until),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            status="active",
            extraction_context=raw.extraction_context,
        )


def _observation_semantic_class(
    projection: SourceProjection,
    observation_id: str | None,
) -> str | None:
    if observation_id is None:
        return None
    for revision in projection.observation_revisions:
        if revision.observation_id != observation_id:
            continue
        value = revision.metadata.get("semantic_class")
        return str(value) if isinstance(value, str) and value else None
    return None


async def _run_concurrently(first: Awaitable[_First], second: Awaitable[_Second]) -> tuple[_First, _Second]:
    """Run two lines at once; the first failure cancels the other and propagates unchanged."""
    tasks = (asyncio.ensure_future(first), asyncio.ensure_future(second))
    try:
        first_result, second_result = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return first_result, second_result


def _support_validation_baseline(support) -> tuple[str | None, str | None]:
    """Return one Evidence Unit's recorded baseline Source Unit revision, or why it is unusable.

    ``(None, None)`` means no baseline was ever recorded for this Support.
    """
    plan_ids = {item.validation_plan_id for item in support}
    baseline_ids = {item.validation_unit_revision_id for item in support}
    if len(plan_ids) != 1 or len(baseline_ids) != 1:
        return None, "validation association is inconsistent"
    baseline_id = next(iter(baseline_ids))
    if next(iter(plan_ids)) is not None and baseline_id is None:
        return None, "validation Plan is not applied to this Source Unit"
    return baseline_id, None


def _projection_evidence_image_loader(projection: SourceProjection, document_store):
    """Load current Artifact Evidence bytes; without an Artifact store none can be supplied.

    Each reader maps missing bytes to its own failure.
    """

    if document_store is None:
        return None
    from memforge.pipeline.projection_images import load_projection_images

    return lambda ids: load_projection_images(projection=projection, observation_ids=ids, document_store=document_store)


def _candidate_rejection_payload(rejection: CandidateRejection, revision: SourceUnitRevision) -> dict[str, Any]:
    """Identify a rejected Candidate and its selected Evidence without Source text."""

    candidate = rejection.candidate
    return {
        "source_unit_id": revision.source_unit_id,
        "target_unit_revision_id": revision.id,
        "candidate_claim": candidate.content,
        "candidate_memory_type": candidate.memory_type,
        "reject_reason": rejection.reject_reason,
        "selected_evidence": [
            {
                "role": part.role.value,
                "kind": part.kind.value,
                "observation_id": part.anchor.observation_id,
                "observation_revision_id": part.anchor.observation_revision_id,
                "range_start": part.anchor.range_start,
                "range_end": part.anchor.range_end,
            }
            for part in candidate.resolved_evidence_selection.parts
        ],
    }


def _source_lifecycle_operation_input_hash(
    *,
    projection: SourceProjection,
    candidates: Sequence[RawMemory],
    incumbents: Sequence[Memory],
    support_hashes: Mapping[str, str],
    gate_state: str,
    update_mode: str,
    changed_hunks: str | None,
    update_plan_stats: Mapping[str, Any] | None,
    llm_model: str,
    input_policy_identity: str | None = None,
) -> str:
    """Digest the exact reconciliation manifest without persisting source content."""

    from memforge.pipeline.claim_revision import CLAIM_REVISION_CONTRACT
    from memforge.pipeline.revision_assessment import REVISION_SUPPORT_CONTRACT, REVISION_INPUT_POLICY

    manifest = {
        "semantic_contract": "/".join((REVISION_SUPPORT_CONTRACT, CANDIDATE_ADMISSION_CONTRACT,
                                       CLAIM_REVISION_CONTRACT, REVISION_INPUT_POLICY,
                                       SPARSE_MEMORY_CLASSIFIER_VERSION)),
        "input_policy_identity": input_policy_identity,
        "projection_identity_hash": source_derivation_projection_identity_hash(projection),
        "candidates": [
            {
                "content_hash": content_hash(candidate.content.strip()),
                "memory_type": candidate.memory_type,
                "confidence": candidate.confidence,
                "source_observation_id": candidate.source_observation_id,
                "required_source_observation_ids": sorted(
                    candidate.required_source_observation_ids
                ),
            }
            for candidate in candidates
        ],
        "incumbents": [
            {
                "memory_id": incumbent.id,
                "memory_version": lifecycle_memory_version(incumbent),
                "support_set_hash": support_hashes.get(incumbent.id),
            }
            for incumbent in sorted(incumbents, key=lambda item: item.id)
        ],
        "gate_state": gate_state,
        "update_mode": update_mode,
        "changed_hunks_hash": (
            hashlib.sha256(changed_hunks.encode("utf-8")).hexdigest()
            if changed_hunks is not None
            else None
        ),
        "update_plan_stats": dict(update_plan_stats or {}),
        "llm_model": llm_model,
    }
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
