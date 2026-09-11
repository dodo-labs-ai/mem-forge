"""Memory extraction and projected-lifecycle orchestration.

The engine is the ownership boundary between source projection/extraction and
durable memory state.  A projected lifecycle call derives its reconciliation
scope, access context, staged evidence, lifecycle plan, and outbox work from one
``SourceProjection`` so those records commit atomically.
"""

from __future__ import annotations

import json
import hashlib

from memforge.llm.structured import StructuredLlmError
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING, Any

from memforge.evals.agent_evaluation import (
    AgentRuntimeBundle,
    NoOpRuntimeEventTraceSink,
    RuntimeEventTraceSink,
    assessment_sink_for_runtime_sink,
    bind_source_lifecycle_outcome,
    current_deployment_revision,
    publish_agent_assessments,
    publish_runtime_events,
)
from memforge.memory.candidate_ledger import (
    CandidateLedgerError,
    CandidateLedgerResult,
    select_unique_memory_candidates,
)
from memforge.memory.entity_resolver import EntityResolver
from memforge.memory.evidence import (
    EvidenceReference,
    EvidenceUnit,
    SupportScopeVersion,
)
from memforge.memory.identity_resolver import (
    IdentityResolutionRequest,
    IdentityResolver,
)
from memforge.memory.lifecycle_plan import (
    LifecycleGateState,
    LifecyclePlan,
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
from memforge.memory.relation_candidate_retrieval import CrossDocumentCandidateRetriever
from memforge.memory.relation_classifier import (
    MEMORY_PAIR_CLASSIFIER_VERSION,
    StructuredMemoryPairClassifier,
)
from memforge.memory.relation_discovery_contract import PreclassifiedRelationDecision
from memforge.source_access import (
    memory_visibility_for_document,
    memory_visibility_for_source_id,
)
from memforge.source_activity import SourceActivityLease
from memforge.source_projection import ImpactResult, ProjectionCoverage, resolve_anchor_impact
from memforge.source_derivation import (
    SourceUnitDerivationContext,
    source_derivation_context_identity_hash,
    source_derivation_projection_identity_hash,
)
from memforge.storage.adapters.protocols import EntityResolutionScope
from memforge.models import (
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
    support_scope_version: SupportScopeVersion
    evidence_reference_ids_by_claim_hash: Mapping[str, tuple[str, ...]]
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
            StructuredMemoryPairClassifier(
                client=structured_llm_client,
                model=llm_model,
            )
            if callable(getattr(structured_llm_client, "classify_memory_relations", None))
            else None
        )
        self.identity_resolver = IdentityResolver(
            memory_store=memory_store,
            pair_classifier=self.pair_classifier,
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
        """Load the complete active ledger by stable Unit, with a legacy fallback.

        A provider-backed rename can change ``doc_id`` without changing the
        Source Unit. Support Assertions are therefore authoritative for the
        projected path; same-document extracted support is included only to
        keep pre-cutover rows visible to the conservative lineage gate.
        """
        support_scope_version = await self.db.get_support_scope_version()
        unit_support = (
            await self.db.get_source_unit_support_unit_ids(source_unit_id)
            if support_scope_version is SupportScopeVersion.EVIDENCE_UNIT_SET_V2
            else await self.db.get_source_unit_support_reference_ids(source_unit_id)
        )
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

        Missing legacy Support, ambiguous mappings, and mixed evidence stay
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
        v2 = (
            await self.db.get_support_scope_version()
            is SupportScopeVersion.EVIDENCE_UNIT_SET_V2
        )
        for memory_id in ordered_incumbent_ids:
            reference_ids = unit_support.get(memory_id)
            if not reference_ids:
                resolved[memory_id] = ImpactResult.UNKNOWN
                continue
            scoped_reference_ids = frozenset(reference_ids)
            evidence = evidence_by_memory_id.get(memory_id, ())
            impacts = {
                resolve_anchor_impact(item.anchor, delta)
                for item in evidence
                if (
                    item.evidence_unit_id in scoped_reference_ids
                    if v2
                    else item.reference_id in scoped_reference_ids
                )
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
        expected_source_activity_epoch: int | None = None,
        source_activity: SourceActivityLease | None = None,
        current_changed_ranges: tuple[tuple[int, int], ...] = (),
        lifecycle_execution_owner_id: str | None = None,
        lifecycle_attempt_count: int = 1,
    ) -> dict[str, int]:
        """Observe one complete Source Unit lifecycle execution."""

        runtime_context = _LifecycleExecutionContext(started_at=perf_counter())
        try:
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
                or (exc.reason_code if isinstance(exc, ReconciliationContractError) else None)
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
                    else (exc.terminal_category if isinstance(exc, StructuredLlmError) else None)
                ),
                error_code=(
                    (exc.reason_code if isinstance(exc, ReconciliationContractError) else
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

        current_support_owners = await self._active_v2_support_owners(
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
            memory_id: support_states[memory_id].support_ids
            for memory_id in memory_ids
        }
        support_hashes = {
            memory_id: support_states[memory_id].support_set_hash
            for memory_id in memory_ids
        }
        source_support = (
            await self.db.get_source_unit_support_unit_ids(
                inputs.scope.source_unit_id
            )
            if inputs.support_scope_version
            is SupportScopeVersion.EVIDENCE_UNIT_SET_V2
            else await self.db.get_source_unit_support_reference_ids(
                inputs.scope.source_unit_id
            )
        )

        return build_lifecycle_plan(
            plan_id=inputs.plan_id,
            scope=inputs.scope,
            gate_state=inputs.gate_state,
            operations=inputs.operations,
            incumbents=current_incumbents,
            source_support_reference_ids=source_support,
            all_active_support_reference_ids=all_support,
            support_set_hashes=support_hashes,
            observation_revision_ids=inputs.observation_revision_ids,
            new_evidence_reference_ids=(),
            evidence_reference_ids_by_claim_hash=(
                inputs.evidence_reference_ids_by_claim_hash
            ),
            support_scope_version=inputs.support_scope_version,
            source_support_unit_ids=(
                source_support
                if inputs.support_scope_version
                is SupportScopeVersion.EVIDENCE_UNIT_SET_V2
                else None
            ),
            all_active_support_unit_ids=(
                all_support if inputs.support_scope_version is SupportScopeVersion.EVIDENCE_UNIT_SET_V2 else None
            ),
            evidence_unit_ids_by_claim_hash=(inputs.evidence_unit_ids_by_claim_hash),
            corroboration_targets_by_claim_hash=current_corroboration_targets,
            corroboration_proofs_by_claim_hash=(
                inputs.corroboration_proofs_by_claim_hash
            ),
            defaults=inputs.defaults,
            evidence_units=inputs.evidence_units,
            evidence_references=inputs.evidence_references,
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
            elif mutation.mutation_type.value == "reactivate_memory":
                stats["reactivated"] += 1
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

    async def _active_v2_support_owners(
        self,
        memory_ids: Sequence[str],
    ) -> dict[str, dict[str, str]]:
        """Return exact active v2 Support ownership for prepared drift guards."""

        owners: dict[str, dict[str, str]] = {
            memory_id: {} for memory_id in memory_ids
        }
        for memory_id in memory_ids:
            for unit in await self.db.get_memory_evidence_units(memory_id):
                if (
                    unit.support_scope_version
                    is not SupportScopeVersion.EVIDENCE_UNIT_SET_V2
                ):
                    continue
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
        expected_source_activity_epoch: int | None = None,
        source_activity: SourceActivityLease | None = None,
        current_changed_ranges: tuple[tuple[int, int], ...] = (),
        lifecycle_execution_owner_id: str | None = None,
        lifecycle_attempt_count: int = 1,
        _runtime_context: _LifecycleExecutionContext,
    ) -> dict[str, int]:
        """Reconcile a complete Source Unit ledger and atomically apply one plan."""

        from memforge.pipeline.reconciler import (
            ReconciliationResult,
            reconcile_memories,
        )
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
            "reactivated": 0,
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
            "support_revalidation_auto_rebind_count": 0,
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
        support_scope_version = await self.db.get_support_scope_version()
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
        _runtime_context.incumbent_count = len(incumbents)
        _runtime_context.stage = "candidate_admission"
        candidate_ledger = await self._select_projected_candidates(
            projection=projection,
            doc_id=doc_id,
            candidates=filtered_memories,
        )
        filtered_memories = list(candidate_ledger.candidates)
        stats.update(
            {
                "candidate_ledger_input_count": candidate_ledger.input_count,
                "candidate_ledger_selected_count": len(candidate_ledger.candidates),
                "candidate_ledger_dropped_exact_count": (candidate_ledger.dropped_exact_count),
                "candidate_ledger_dropped_redundant_count": (candidate_ledger.dropped_redundant_count),
                "candidate_ledger_dropped_low_value_count": (candidate_ledger.dropped_low_value_count),
                "candidate_ledger_llm_calls": candidate_ledger.structured_llm_calls,
                "candidate_ledger_llm_elapsed_ms": (candidate_ledger.structured_llm_elapsed_ms),
                "candidate_ledger_validation_retries": (candidate_ledger.validation_retries),
                "candidate_ledger_fallback_batch_count": (candidate_ledger.fallback_batch_count),
                "candidate_ledger_fallback_candidate_count": (candidate_ledger.fallback_candidate_count),
                "candidate_ledger_prompt_chars": candidate_ledger.prompt_chars,
            }
        )
        stats["skipped"] += quality_candidate_count - len(filtered_memories)
        _runtime_context.model_call_count += candidate_ledger.structured_llm_calls
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
        support_audits = []
        skipped_revalidation: dict[str, str] = {}
        assessed_evidence: dict[str, list[RawMemory]] = {}
        assessment_image_loader = None
        required_derivation_work_ids = ()
        model_incumbent_count = 0
        deterministic_disjoint_keep_count = 0
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
            deterministic_disjoint_ids = frozenset()
            model_incumbents = [
                memory
                for memory in incumbents
                if memory.id not in (deterministic_disjoint_ids | derivation_protected_ids)
            ]
            model_incumbent_count = len(model_incumbents)
            deterministic_disjoint_keep_count = len(deterministic_disjoint_ids)
            if model_incumbents and not self.structured_llm_client:
                raise RuntimeError("complete lifecycle reconciliation requires an LLM client")
            if model_incumbents:
                from memforge.pipeline.revision_assessment import RevisionAssessmentContext
                from memforge.pipeline.projection_images import load_projection_images, projection_inference_image_observation_ids
                from memforge.pipeline.reconciler import SupportAuditEntry, ReconciliationContractError

                def assessment_image_loader(ids):
                    if self.document_store is None:
                        raise SupportRevalidationLimitation(
                            SupportRevalidationLimitationCode.UNSUPPORTED_REPRESENTATION,
                            "revision assessment requires the Artifact store",
                        )
                    return load_projection_images(
                        projection=projection, observation_ids=ids, document_store=self.document_store,
                    )
                base = await self.db.get_current_source_unit_projection(scope.source_unit_id)
                if base is not None and base.source_unit_revisions[0].id not in {scope.base_unit_revision_id, scope.target_unit_revision_id}:
                    raise AuthorityPlanStaleError("revision assessment base changed")
                assessment_context = RevisionAssessmentContext(
                    projection=projection, base=base, access_context_hash=access_context_hash, image_loader=assessment_image_loader,
                )
                contexts_by_revision = {base.source_unit_revisions[0].id: assessment_context} if base else {}
                evidence_by_memory = await self.db.get_active_memory_support_evidence_many(
                    tuple(memory.id for memory in model_incumbents), source_id=projection.source_id,
                )
                from memforge.pipeline.revision_work import RevisionWorkExecutor, SupportWorkItem
                work_items = []
                work_by_memory = {}
                for memory in model_incumbents:
                    groups: dict[str, list] = {}
                    for item in evidence_by_memory.get(memory.id, ()):
                        scoped_id = item.evidence_unit_id if support_scope_version is SupportScopeVersion.EVIDENCE_UNIT_SET_V2 else item.reference_id
                        if scoped_id in unit_support.get(memory.id, ()):
                            groups.setdefault(item.evidence_unit_id, []).append(item)
                    if not groups:
                        raise ReconciliationContractError("revision_support_missing", "incumbent has no complete scoped support")
                    work_by_memory[memory.id] = []
                    for evidence_unit_id, support in groups.items():
                        plan_ids = {item.validation_plan_id for item in support}
                        baseline_ids = {item.validation_unit_revision_id for item in support}
                        if len(plan_ids) != 1 or len(baseline_ids) != 1:
                            raise ReconciliationContractError("revision_support_baseline_invalid", "Support validation association is inconsistent")
                        baseline_id = next(iter(baseline_ids))
                        if next(iter(plan_ids)) is not None and baseline_id is None:
                            raise ReconciliationContractError("revision_support_baseline_invalid", "Support validation Plan is not applied to this Source Unit")
                        context = contexts_by_revision.get(baseline_id)
                        if context is None:
                            historical = (
                                await self.db.get_source_unit_revision_projection(scope.source_unit_id, baseline_id)
                                if baseline_id is not None else None
                            )
                            if baseline_id is not None and (
                                historical is None
                                or historical.source_id != projection.source_id
                                or len(historical.source_unit_revisions) != 1
                                or historical.source_unit_revisions[0].source_unit_id != scope.source_unit_id
                                or historical.source_unit_revisions[0].id != baseline_id
                            ):
                                raise ReconciliationContractError("revision_support_baseline_invalid", "Support validation snapshot is unavailable or inconsistent")
                            context = RevisionAssessmentContext(
                                projection=projection, base=historical, access_context_hash=access_context_hash,
                                image_loader=assessment_image_loader, indexes=assessment_context.indexes,
                            )
                            contexts_by_revision[baseline_id] = context
                        work_id = f"w{len(work_items):06d}"
                        work_items.append(SupportWorkItem(work_id, memory, tuple(support), context))
                        work_by_memory[memory.id].append(work_id)
                evaluator = RevisionWorkExecutor(client=self.structured_llm_client, model=self.llm_model,
                                                 store=self.db, derivation_id=derivation_id)
                stats["support_revalidation_work_item_count"] += len(work_items)
                try:
                    _runtime_context.stage = "support_revalidation"
                    assessed = await evaluator.assess_many(work_items)
                finally:
                    stats["support_revalidation_model_call_count"] += evaluator.calls
                    stats["support_revalidation_assessment_request_count"] = evaluator.stage_counts["support_assess"]
                    stats["support_revalidation_prompt_chars"] += evaluator.prompt_chars
                    stats["support_revalidation_reused_work_count"] = evaluator.reused
                    stats["support_revalidation_covered_source_claim_pairs"] = evaluator.covered_source_claim_pairs
                    _runtime_context.model_call_count += evaluator.calls
                stats["support_revalidation_completion_count"] = len(evaluator.final_work_ids)
                required_derivation_work_ids = tuple(evaluator.final_work_ids) if derivation_id else ()
                for memory in model_incumbents:
                    results = [assessed[work_id] for work_id in work_by_memory[memory.id]]
                    if any(result.supported is None for result in results):
                        skipped_revalidation[memory.id] = "; ".join(
                            dict.fromkeys(result.reason for result in results if result.supported is None)
                        )
                        continue
                    current = [result.memory for result in results if result.supported and result.memory is not None]
                    stats["support_revalidation_auto_rebind_count"] += len(current)
                    assessed_evidence[memory.id] = current
                    support_audits.append(SupportAuditEntry(
                        incumbent_id=memory.id, supported=bool(current),
                        reason="; ".join(dict.fromkeys(result.reason for result in results)),
                    ))
                stats["support_revalidation_revision_index_count"] = len(assessment_context.indexes)
            _runtime_context.stage = "reconciliation"
            result = await reconcile_memories(
                new_extractions=filtered_memories,
                existing_memories=[memory for memory in model_incumbents if memory.id not in skipped_revalidation],
                doc_type=doc_type,
                structured_llm_client=self.structured_llm_client,
                llm_model=self.llm_model,
                updated_document=document_content,
                update_mode=update_mode,
                changed_hunks=changed_hunks,
                update_plan_stats=update_plan_stats,
                include_metadata=True,
                support_audits=support_audits,
                image_loader=assessment_image_loader,
                work_store=self.db,
                derivation_id=derivation_id,
                operation_input_hash=operation_input_hash,
            )
            if not isinstance(result, ReconciliationResult):
                raise TypeError("metadata reconciliation must return ReconciliationResult")
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
                    action=ReconcileAction.NOOP,
                    memory_id=memory_id,
                    reason=reason or "Support revalidation insufficient; preserve existing evidence",
                    support_revalidation_skipped=True,
                )
                for memory_id, reason in skipped_revalidation.items()
            )
            skipped_revalidation.update({
                operation.memory_id: operation.reason
                for operation in result.operations
                if operation.support_revalidation_skipped and operation.memory_id
            })
            stats["support_revalidation_skipped_memory_count"] = len(skipped_revalidation)
            operations += tuple(
                ReconcileOperation(
                    action=ReconcileAction.DELETE,
                    memory_id=memory_id,
                    reason=("current Source Artifact revision is not inference eligible"),
                    flag_for_review=True,
                )
                for memory_id in sorted(derivation_protected_ids)
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
                    "reconciliation_disjoint_keep_count": (deterministic_disjoint_keep_count),
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
        for operation in operations:
            if (
                operation.action
                not in {
                    ReconcileAction.ADD,
                    ReconcileAction.UPDATE,
                    ReconcileAction.SUPERSEDE,
                }
                or operation.memory is None
            ):
                continue
            quality = classify_memory_candidate(operation.memory)
            if not quality.keep:
                raise RuntimeError(
                    "complete lifecycle reconciliation produced an unsafe Memory candidate: "
                    f"{quality.skip_reason or 'quality_rejected'}"
                )
        incumbents_by_id = {memory.id: memory for memory in incumbents}
        operations = tuple(
            replace(operation, memory=assessed_evidence[operation.memory_id][0])
            if (operation.action is ReconcileAction.NOOP and not operation.support_revalidation_skipped
                and assessed_evidence.get(operation.memory_id))
            else operation
            for operation in operations
        )
        _runtime_context.stage = "plan_construction"
        corroboration_targets: dict[str, Memory] = {}
        corroboration_proofs: dict[str, dict[str, object]] = {}
        preclassified_relations: dict[str, tuple[PreclassifiedRelationDecision, ...]] = {}
        identity_claim_hashes: list[str] = []
        identity_requests: list[IdentityResolutionRequest] = []
        operation_memories = tuple(operation.memory for operation in operations if operation.memory is not None)
        entity_resolution = await self.entity_resolver.resolve_many(
            tuple(entity_ref for raw_memory in operation_memories for entity_ref in raw_memory.entity_refs),
            scope=EntityResolutionScope(access_context_hash=access_context_hash),
            doc_context=document_content[:2000],
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
                    excluded_memory_ids=frozenset(incumbents_by_id).difference(skipped_revalidation),
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
        classified_candidate_ids = tuple(
            dict.fromkeys(
                memory_id
                for resolution in identity_resolutions
                for memory_id in (
                    *(decision.candidate_memory_id for decision in resolution.classified_pairs),
                    *((resolution.target.id,) if resolution.target is not None else ()),
                )
            )
        )
        classified_candidate_support = await self.db.get_active_memory_support_states(classified_candidate_ids)
        attached_target_ids: list[str] = []
        for claim_hash, resolution in zip(
            identity_claim_hashes,
            identity_resolutions,
            strict=True,
        ):
            target = resolution.target
            equivalence_proof = resolution.equivalence_proof
            preclassified_relations[claim_hash] = tuple(
                PreclassifiedRelationDecision(
                    candidate_memory_id=decision.candidate_memory_id,
                    expected_candidate_content_hash=decision.candidate_content_hash,
                    expected_candidate_support_set_hash=(
                        classified_candidate_support[decision.candidate_memory_id].current_support_set_hash
                    ),
                    expected_candidate_access_context_hash=lifecycle_access_context_hash(
                        visibility=decision.candidate_visibility,
                        owner_user_id=decision.candidate_owner_user_id,
                        project_key=decision.candidate_project_key,
                        repo_identifier=decision.candidate_repo_identifier,
                    ),
                    expected_challenger_access_context_hash=access_context_hash,
                    relation_type=decision.relation_type,
                    direction=decision.direction,
                    reason=decision.reason,
                    classifier_version=MEMORY_PAIR_CLASSIFIER_VERSION,
                )
                for decision in resolution.classified_pairs
            )
            if target is None or equivalence_proof is None:
                continue
            corroboration_targets[claim_hash] = target
            corroboration_proofs[claim_hash] = dict(equivalence_proof)
            attached_target_ids.append(target.id)
        evidence_memories = [operation.memory for operation in operations if operation.memory is not None]
        for operation in operations:
            if operation.action is ReconcileAction.NOOP:
                evidence_memories.extend(assessed_evidence.get(operation.memory_id, [])[1:])
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
            support_scope_version=support_scope_version,
        )
        operations = tuple(
            replace(
                operation,
                memory=projected_evidence.canonical_memories_by_claim_hash[
                    content_hash(operation.memory.content.strip())
                ],
            )
            if operation.memory is not None
            else operation
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
            preclassified_relations_by_claim_hash=preclassified_relations,
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
        initial_support_owners = await self._active_v2_support_owners(
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
                support_scope_version=support_scope_version,
                evidence_reference_ids_by_claim_hash=(
                    projected_evidence.reference_ids_by_claim_hash
                ),
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
                candidate_ledger.structured_llm_calls
                + structured_llm_call_count
                + int(stats["support_revalidation_model_call_count"])
                + entity_resolution.metrics.structured_llm_calls
                + identity_resolution.metrics.llm_calls
            ),
            prepared_at_attempt_count=lifecycle_attempt_count,
        )
        _runtime_context.stage = "lifecycle_commit"
        return await self._commit_prepared_projected_lifecycle(
            prepared,
            lifecycle_attempt_count=lifecycle_attempt_count,
        )

    async def _select_projected_candidates(
        self,
        *,
        projection: SourceProjection,
        doc_id: str,
        candidates: list[RawMemory],
    ) -> CandidateLedgerResult:
        """Select bounded within-revision candidate admission before writes."""

        try:
            result = await select_unique_memory_candidates(
                candidates,
                structured_llm_client=self.structured_llm_client,
                llm_model=self.llm_model,
            )
        except CandidateLedgerError as exc:
            await self._record_candidate_ledger_audit(
                projection=projection,
                doc_id=doc_id,
                status="failed",
                reason=exc.error_type,
                payload={
                    "input_count": exc.input_count,
                    "semantic_input_count": exc.semantic_input_count,
                    "selected_count": 0,
                    "structured_llm_calls": exc.structured_llm_calls,
                    "structured_llm_elapsed_ms": exc.structured_llm_elapsed_ms,
                    "validation_retries": exc.validation_retries,
                    "prompt_chars": exc.prompt_chars,
                    "candidate_fingerprints": _candidate_fingerprints(candidates),
                    "fingerprints_truncated": len(candidates) > 200,
                },
                error=str(exc),
            )
            raise RuntimeError(f"candidate ledger failed closed: {exc.error_type}: {exc}") from exc

        if result.semantic_input_count > 1 or result.dropped_exact_count:
            await self._record_candidate_ledger_audit(
                projection=projection,
                doc_id=doc_id,
                status="committed",
                reason=(
                    "candidate_admission_with_fallback" if result.fallback_batch_count else "complete_candidate_ledger"
                ),
                payload=_candidate_ledger_audit_payload(result),
            )
        return result

    async def _record_candidate_ledger_audit(
        self,
        *,
        projection: SourceProjection,
        doc_id: str,
        status: str,
        reason: str,
        payload: dict[str, Any],
        error: str | None = None,
    ) -> None:
        context = self.memory_store.operation_context(
            run_id=projection.run_id,
            source_id=projection.source_id,
            doc_id=doc_id,
        )
        await self.memory_store.record_audit_event(
            "candidate_ledger_completed" if status == "committed" else "candidate_ledger_failed",
            status,
            context=context,
            doc_id=doc_id,
            source_id=projection.source_id,
            decision="select_unique_candidates",
            reason=reason,
            payload=payload,
            error=error,
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
        support_scope_version = await self.db.get_support_scope_version()
        all_support = {memory_id: state.support_ids for memory_id, state in support_states.items()}
        support_hashes = {memory_id: state.support_set_hash for memory_id, state in support_states.items()}
        visibility, owner_user_id = await memory_visibility_for_document(self.db, doc_id=doc_id)
        plan = build_lifecycle_plan(
            plan_id=plan_id,
            scope=scope,
            gate_state=gate.state,
            operations=operations,
            incumbents=incumbents_by_id,
            source_support_reference_ids=unit_support,
            all_active_support_reference_ids=all_support,
            support_set_hashes=support_hashes,
            observation_revision_ids=(),
            new_evidence_reference_ids=(),
            support_scope_version=support_scope_version,
            source_support_unit_ids=(
                unit_support
                if support_scope_version is SupportScopeVersion.EVIDENCE_UNIT_SET_V2
                else None
            ),
            all_active_support_unit_ids=(
                all_support
                if support_scope_version is SupportScopeVersion.EVIDENCE_UNIT_SET_V2
                else None
            ),
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
            contradiction_count=0,
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


def _candidate_ledger_audit_payload(result: CandidateLedgerResult) -> dict[str, Any]:
    return {
        "input_count": result.input_count,
        "semantic_input_count": result.semantic_input_count,
        "selected_count": len(result.candidates),
        "dropped_exact_count": result.dropped_exact_count,
        "dropped_redundant_count": result.dropped_redundant_count,
        "dropped_low_value_count": result.dropped_low_value_count,
        "structured_llm_calls": result.structured_llm_calls,
        "structured_llm_elapsed_ms": result.structured_llm_elapsed_ms,
        "validation_retries": result.validation_retries,
        "fallback_batch_count": result.fallback_batch_count,
        "fallback_candidate_count": result.fallback_candidate_count,
        "prompt_chars": result.prompt_chars,
        "drops": [
            {
                "candidate_content_hash": content_hash(drop.candidate.content),
                "candidate_source_observation_id": drop.candidate.source_observation_id,
                "canonical_content_hash": (
                    content_hash(drop.canonical_candidate.content) if drop.canonical_candidate is not None else None
                ),
                "canonical_source_observation_id": (
                    drop.canonical_candidate.source_observation_id if drop.canonical_candidate is not None else None
                ),
                "method": drop.method,
                "reason": drop.reason[:240],
            }
            for drop in result.drops
        ],
    }


def _candidate_fingerprints(
    candidates: list[RawMemory],
    *,
    limit: int = 200,
) -> list[dict[str, str | None]]:
    return [
        {
            "content_hash": content_hash(candidate.content),
            "source_observation_id": candidate.source_observation_id,
        }
        for candidate in candidates[:limit]
    ]


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
        "semantic_contract": "/".join((REVISION_SUPPORT_CONTRACT, CLAIM_REVISION_CONTRACT,
                                       REVISION_INPUT_POLICY, MEMORY_PAIR_CLASSIFIER_VERSION)),
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
