"""Provider-neutral, side-effect-free offline agent evaluation.

The module is intentionally a deep boundary: callers curate immutable inputs,
accept a reference, freeze a cohort, execute a run, and read a report. Storage,
identity, replay dispatch, caching, and deterministic scoring stay internal.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from time import perf_counter
from typing import Protocol

from memforge.evals.agent_evaluation import AgentAssessment, AgentAssessmentLabel
from memforge.models import Memory, MemoryExtractionResult, RawMemory, ReconcileOperation
from memforge.pipeline.reconciler import ReconciliationResult, reconcile_memories
from memforge.source_derivation import (
    SourceDerivationBatch,
    SourceUnitDerivationRequest,
    memory_extraction_output_payload,
    replay_source_unit_derivation,
    source_unit_derivation_context_from_payload,
)
from memforge.source_projection import source_projection_from_payload


OFFLINE_EVALUATION_SCHEMA_VERSION = "1"
OFFLINE_CONTENT_POLICY_SCHEMA_VERSION = "1"
ACCEPTED_GROUND_TRUTH_SCHEMA_VERSION = "2"
OFFLINE_DETERMINISTIC_EVALUATOR_VERSION = "1"


class AgentEvaluationCaseKind(str, Enum):
    SOURCE_UNIT_DERIVATION = "source_unit_derivation_v1"
    SOURCE_UNIT_RECONCILIATION = "source_unit_reconciliation_v1"


class AgentEvaluationPopulation(str, Enum):
    FAILURE_REGRESSION = "failure_regression"
    REPRESENTATIVE_CONTROL = "representative_control"


class AgentEvaluationRole(str, Enum):
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    RELEASE_HOLDOUT = "release_holdout"
    SENTINEL = "sentinel"


class AgentEvaluationRunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentEvaluationResultStatus(str, Enum):
    COMPLETED = "completed"
    ERROR = "error"
    ARTIFACT_UNAVAILABLE = "artifact_unavailable"


class AgentEvaluationContentProfile(str, Enum):
    """Fixed disclosure profiles; profiles expand only through a new version."""

    HUMAN_CALIBRATION = "human_calibration_v1"


class DeterministicCheckLabel(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AgentEvaluationCase:
    case_id: str
    case_kind: AgentEvaluationCaseKind
    source_id: str
    doc_id: str
    source_unit_id: str
    manifest: Mapping[str, object]
    manifest_hash: str
    promotion_policy_version: str
    created_by: str
    created_at: str
    source_runtime_event_id: str | None = None
    supersedes_case_id: str | None = None
    schema_version: str = OFFLINE_EVALUATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class AgentEvaluationContentPolicy:
    """One immutable approval to disclose protected evaluation content."""

    content_policy_id: str
    source_id: str
    profile: AgentEvaluationContentProfile
    policy_version: str
    approved_by: str
    approved_at: str
    schema_version: str = OFFLINE_CONTENT_POLICY_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class AgentEvaluationAnnotationTask:
    """Blinded case and candidate content presented to one authorized reviewer."""

    result_id: str
    case_id: str
    case_kind: AgentEvaluationCaseKind
    content_policy_id: str
    case_manifest: Mapping[str, object]
    candidate_output: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class AcceptedGroundTruthRevision:
    ground_truth_revision_id: str
    case_id: str
    rubric: Mapping[str, object]
    rubric_hash: str
    accepted_by: str
    accepted_at: str
    supporting_assessment_ids: tuple[str, ...] = ()
    acceptance_policy_version: str | None = None
    adjudication_note: str | None = None
    schema_version: str = ACCEPTED_GROUND_TRUTH_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class AgentEvaluationCohortItem:
    case_id: str
    ground_truth_revision_id: str
    population: AgentEvaluationPopulation
    role: AgentEvaluationRole
    group_key: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.group_key:
            raise ValueError("cohort item requires group_key")
        if self.weight <= 0:
            raise ValueError("cohort item weight must be positive")


@dataclass(frozen=True, slots=True)
class AgentEvaluationCohort:
    cohort_id: str
    items: tuple[AgentEvaluationCohortItem, ...]
    selection_policy_version: str
    manifest_hash: str
    created_by: str
    created_at: str
    schema_version: str = OFFLINE_EVALUATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class AgentEvaluationRun:
    run_id: str
    cohort_id: str
    candidate_manifest: Mapping[str, object]
    candidate_manifest_hash: str
    evaluator_suite: str
    evaluator_version: str
    replicate_count: int
    status: AgentEvaluationRunStatus
    created_by: str
    created_at: str
    completed_at: str | None = None
    baseline_run_id: str | None = None
    schema_version: str = OFFLINE_EVALUATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class DeterministicCheck:
    criterion: str
    label: DeterministicCheckLabel
    reason_code: str


@dataclass(frozen=True, slots=True)
class AgentEvaluationResult:
    result_id: str
    run_id: str
    case_id: str
    ground_truth_revision_id: str
    replicate_ordinal: int
    candidate_output_key: str
    status: AgentEvaluationResultStatus
    output: Mapping[str, object] | None
    output_hash: str | None
    duration_ms: int
    created_at: str
    error_code: str | None = None
    reused_from_result_id: str | None = None
    schema_version: str = OFFLINE_EVALUATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class AgentEvaluationRunReport:
    run: AgentEvaluationRun
    results: tuple[AgentEvaluationResult, ...]
    assessments: tuple[AgentAssessment, ...]
    completed_result_count: int
    error_result_count: int
    check_counts: Mapping[str, int]
    population_summaries: Mapping[str, Mapping[str, int]]


class OfflineEvaluationStore(Protocol):
    async def authorize_agent_evaluation_source(
        self,
        source_id: str,
        actor_user_id: str,
    ) -> None: ...

    async def record_agent_evaluation_case(self, case: AgentEvaluationCase) -> None: ...
    async def get_agent_evaluation_case(self, case_id: str) -> AgentEvaluationCase | None: ...
    async def record_agent_evaluation_content_policy(
        self, policy: AgentEvaluationContentPolicy
    ) -> None: ...
    async def get_agent_evaluation_content_policy(
        self, content_policy_id: str
    ) -> AgentEvaluationContentPolicy | None: ...
    async def record_accepted_ground_truth_revision(
        self, revision: AcceptedGroundTruthRevision
    ) -> None: ...
    async def get_accepted_ground_truth_revision(
        self, revision_id: str
    ) -> AcceptedGroundTruthRevision | None: ...
    async def record_agent_evaluation_cohort(self, cohort: AgentEvaluationCohort) -> None: ...
    async def get_agent_evaluation_cohort(
        self, cohort_id: str
    ) -> AgentEvaluationCohort | None: ...
    async def record_agent_evaluation_run(self, run: AgentEvaluationRun) -> None: ...
    async def get_agent_evaluation_run(self, run_id: str) -> AgentEvaluationRun | None: ...
    async def record_agent_evaluation_result(
        self,
        result: AgentEvaluationResult,
        assessments: tuple[AgentAssessment, ...] = (),
    ) -> None: ...
    async def get_agent_evaluation_result(
        self, result_id: str
    ) -> AgentEvaluationResult | None: ...
    async def get_cached_agent_evaluation_output(
        self,
        candidate_output_key: str,
    ) -> AgentEvaluationResult | None: ...
    async def list_agent_evaluation_results(
        self, run_id: str
    ) -> list[AgentEvaluationResult]: ...
    async def list_agent_assessments_for_run(self, run_id: str) -> list[AgentAssessment]: ...
    async def record_agent_assessments(
        self, assessments: tuple[AgentAssessment, ...]
    ) -> None: ...
    async def get_agent_assessment(self, assessment_id: str) -> AgentAssessment | None: ...
    async def list_agent_assessments_for_result(
        self, result_id: str
    ) -> list[AgentAssessment]: ...


class OfflineCaseExecutor(Protocol):
    async def execute(
        self,
        case: AgentEvaluationCase,
        candidate_manifest: Mapping[str, object],
    ) -> Mapping[str, object]: ...


class SourceUnitDerivationReplayExecutor:
    """Replay the production Source Unit derivation planner without staging."""

    def __init__(
        self,
        extract_batch: Callable[
            [SourceDerivationBatch, Mapping[str, object]],
            Awaitable[MemoryExtractionResult],
        ],
    ) -> None:
        self._extract_batch = extract_batch

    async def execute(
        self,
        case: AgentEvaluationCase,
        candidate_manifest: Mapping[str, object],
    ) -> Mapping[str, object]:
        projection_payload = _mapping(case.manifest, "projection")
        context_payload = _mapping(case.manifest, "context")
        projection = source_projection_from_payload(projection_payload)
        context = source_unit_derivation_context_from_payload(context_payload)

        async def extract(batch: SourceDerivationBatch) -> MemoryExtractionResult:
            return await self._extract_batch(batch, candidate_manifest)

        result = await replay_source_unit_derivation(
            SourceUnitDerivationRequest(
                projection=projection,
                context=context,
                extract_batch=extract,
                max_concurrent=_positive_int(candidate_manifest.get("max_concurrent"), default=1),
            )
        )
        return {
            "case_kind": case.case_kind.value,
            "extraction": memory_extraction_output_payload(result),
            "error_type": result.error_type,
        }


class SourceUnitReconciliationReplayExecutor:
    """Replay relation classification/reduction without applying a lifecycle plan."""

    def __init__(self, structured_llm_client: object) -> None:
        self._structured_llm_client = structured_llm_client

    async def execute(
        self,
        case: AgentEvaluationCase,
        candidate_manifest: Mapping[str, object],
    ) -> Mapping[str, object]:
        manifest = case.manifest
        new_extractions = [
            _raw_memory_from_payload(item)
            for item in _mapping_list(manifest, "new_extractions")
        ]
        incumbents = [
            _memory_from_payload(item) for item in _mapping_list(manifest, "incumbents")
        ]
        result = await reconcile_memories(
            new_extractions=new_extractions,
            existing_memories=incumbents,
            doc_type=str(manifest.get("doc_type") or "document"),
            structured_llm_client=self._structured_llm_client,
            llm_model=str(candidate_manifest.get("model") or manifest.get("model") or ""),
            updated_document=_optional_str(manifest.get("updated_document")),
            update_mode=str(manifest.get("update_mode") or "full_document"),
            changed_hunks=_optional_str(manifest.get("changed_hunks")),
            update_plan_stats=(
                dict(value) if isinstance((value := manifest.get("update_plan_stats")), Mapping) else None
            ),
            include_metadata=True,
        )
        if not isinstance(result, ReconciliationResult):
            raise TypeError("offline reconciliation requires metadata result")
        return {
            "case_kind": case.case_kind.value,
            "operations": [_reconcile_operation_payload(operation) for operation in result.operations],
            "failure": (
                {
                    "error_type": result.failure.error_type,
                    "reason_code": result.failure.reason_code,
                }
                if result.failure is not None
                else None
            ),
            "metrics": {
                "relation_pair_count": result.metrics.relation_pair_count,
                "revision_proof_count": result.metrics.revision_proof_count,
                "revision_proof_failure_count": result.metrics.revision_proof_failure_count,
            },
        }


class OfflineAgentEvaluation:
    """Application service for the five offline-evaluation operations."""

    def __init__(
        self,
        store: OfflineEvaluationStore,
        *,
        executors: Mapping[AgentEvaluationCaseKind, OfflineCaseExecutor],
    ) -> None:
        self._store = store
        self._executors = dict(executors)

    async def curate_case(
        self,
        *,
        case_kind: AgentEvaluationCaseKind,
        source_id: str,
        doc_id: str,
        source_unit_id: str,
        manifest: Mapping[str, object],
        promotion_policy_version: str,
        created_by: str,
        source_runtime_event_id: str | None = None,
        supersedes_case_id: str | None = None,
        created_at: str | None = None,
    ) -> AgentEvaluationCase:
        await self._store.authorize_agent_evaluation_source(source_id, created_by)
        replay_manifest = {
            **manifest,
            "lineage": {
                "source_id": source_id,
                "doc_id": doc_id,
                "source_unit_id": source_unit_id,
            },
        }
        _validate_case_manifest(case_kind, replay_manifest)
        manifest_hash = _hash(replay_manifest)
        case_id = _id(
            "aec",
            OFFLINE_EVALUATION_SCHEMA_VERSION,
            case_kind.value,
            source_id,
            doc_id,
            source_unit_id,
            manifest_hash,
            promotion_policy_version,
        )
        existing = await self._store.get_agent_evaluation_case(case_id)
        if existing is not None:
            return existing
        case = AgentEvaluationCase(
            case_id=case_id,
            case_kind=case_kind,
            source_id=_required("source_id", source_id),
            doc_id=_required("doc_id", doc_id),
            source_unit_id=_required("source_unit_id", source_unit_id),
            manifest=_canonical_mapping(replay_manifest),
            manifest_hash=manifest_hash,
            promotion_policy_version=_required(
                "promotion_policy_version", promotion_policy_version
            ),
            created_by=_required("created_by", created_by),
            created_at=created_at or _now(),
            source_runtime_event_id=source_runtime_event_id,
            supersedes_case_id=supersedes_case_id,
        )
        await self._store.record_agent_evaluation_case(case)
        return case

    async def approve_human_calibration_content(
        self,
        *,
        source_id: str,
        policy_version: str,
        approved_by: str,
        approved_at: str | None = None,
    ) -> AgentEvaluationContentPolicy:
        """Approve the one bounded Slice-B protected-content profile."""

        await self._store.authorize_agent_evaluation_source(source_id, approved_by)
        version = _required("policy_version", policy_version)
        content_policy_id = _id(
            "aep",
            source_id,
            AgentEvaluationContentProfile.HUMAN_CALIBRATION.value,
            version,
        )
        existing = await self._store.get_agent_evaluation_content_policy(content_policy_id)
        if existing is not None:
            return existing
        policy = AgentEvaluationContentPolicy(
            content_policy_id=content_policy_id,
            source_id=_required("source_id", source_id),
            profile=AgentEvaluationContentProfile.HUMAN_CALIBRATION,
            policy_version=version,
            approved_by=_required("approved_by", approved_by),
            approved_at=approved_at or _now(),
        )
        await self._store.record_agent_evaluation_content_policy(policy)
        return policy

    async def prepare_human_annotation(
        self,
        *,
        result_id: str,
        content_policy_id: str,
        reviewer_id: str,
    ) -> AgentEvaluationAnnotationTask:
        """Return protected content only after source and policy authorization."""

        result, case, policy = await self._human_annotation_context(
            result_id=result_id,
            content_policy_id=content_policy_id,
            reviewer_id=reviewer_id,
        )
        if result.output is None:
            raise ValueError("human annotation requires a completed candidate output")
        return AgentEvaluationAnnotationTask(
            result_id=result.result_id,
            case_id=case.case_id,
            case_kind=case.case_kind,
            content_policy_id=policy.content_policy_id,
            case_manifest=_canonical_mapping(case.manifest),
            candidate_output=_canonical_mapping(result.output),
        )

    async def record_human_annotation(
        self,
        *,
        result_id: str,
        content_policy_id: str,
        criterion: str,
        label: AgentAssessmentLabel,
        reason_code: str,
        rubric_version: str,
        reviewer_id: str,
        created_at: str | None = None,
    ) -> AgentAssessment:
        """Append one immutable human judgment without exposing peer labels."""

        result, _case, policy = await self._human_annotation_context(
            result_id=result_id,
            content_policy_id=content_policy_id,
            reviewer_id=reviewer_id,
        )
        criterion = _required("criterion", criterion)
        reason_code = _required("reason_code", reason_code)
        rubric_version = _required("rubric_version", rubric_version)
        reviewer_id = _required("reviewer_id", reviewer_id)
        if label not in {"pass", "fail", "needs_review"}:
            raise ValueError("human annotation label must be pass, fail, or needs_review")
        assessment = AgentAssessment(
            assessment_id=_id(
                "aas",
                result.result_id,
                criterion,
                "human",
                reviewer_id,
                rubric_version,
                label,
                reason_code,
                policy.content_policy_id,
            ),
            target_event_id=None,
            target_result_id=result.result_id,
            criterion=criterion,
            status="completed",
            label=label,
            reason_code=reason_code,
            annotator_kind="human",
            evaluator_name="memforge.human.calibration",
            evaluator_version=rubric_version,
            annotator_id=reviewer_id,
            content_policy_id=policy.content_policy_id,
            created_at=datetime.fromisoformat(created_at or _now()),
        )
        existing = await self._store.get_agent_assessment(assessment.assessment_id)
        if existing is not None:
            return existing
        await self._store.record_agent_assessments((assessment,))
        return assessment

    async def adjudicate_ground_truth(
        self,
        *,
        case_id: str,
        supporting_assessment_ids: Sequence[str],
        rubric: Mapping[str, object],
        acceptance_policy_version: str,
        adjudication_note: str,
        accepted_by: str,
        accepted_at: str | None = None,
    ) -> AcceptedGroundTruthRevision:
        """Accept a reference while preserving two independent human labels."""

        case = await self._require_case(case_id)
        await self._store.authorize_agent_evaluation_source(case.source_id, accepted_by)
        assessment_ids = tuple(sorted(set(supporting_assessment_ids)))
        if len(assessment_ids) != 2:
            raise ValueError("ground-truth adjudication requires exactly two annotations")
        assessments = []
        for assessment_id in assessment_ids:
            assessment = await self._store.get_agent_assessment(assessment_id)
            if assessment is None:
                raise KeyError(f"unknown human annotation: {assessment_id}")
            assessments.append(assessment)
        if any(
            assessment.annotator_kind != "human"
            or assessment.status != "completed"
            or assessment.target_result_id is None
            or assessment.annotator_id is None
            or assessment.content_policy_id is None
            for assessment in assessments
        ):
            raise ValueError("adjudication accepts only completed human annotations")
        if len({assessment.annotator_id for assessment in assessments}) != 2:
            raise ValueError("adjudication requires two independent reviewers")
        content_policy_ids = {
            assessment.content_policy_id
            for assessment in assessments
            if assessment.content_policy_id is not None
        }
        if len(content_policy_ids) != 1:
            raise ValueError("adjudication annotations must use the same content policy")
        if len({assessment.target_result_id for assessment in assessments}) != 1:
            raise ValueError("adjudication annotations must target the same result")
        if len({assessment.criterion for assessment in assessments}) != 1:
            raise ValueError("adjudication annotations must use the same criterion")
        if len({assessment.evaluator_version for assessment in assessments}) != 1:
            raise ValueError("adjudication annotations must use the same rubric version")
        [target_result_id] = {
            assessment.target_result_id for assessment in assessments if assessment.target_result_id
        }
        [content_policy_id] = content_policy_ids
        _target_result, target_case, _content_policy = await self._human_annotation_context(
            result_id=target_result_id,
            content_policy_id=content_policy_id,
            reviewer_id=accepted_by,
        )
        if target_case.case_id != case.case_id:
            raise ValueError("adjudication annotations do not belong to the requested case")
        note = adjudication_note.strip()
        if not note:
            raise ValueError("adjudication_note is required")
        if len(note) > 2000:
            raise ValueError("adjudication_note must not exceed 2000 characters")
        _validate_rubric(case.case_kind, rubric)
        rubric_payload = _canonical_mapping(rubric)
        rubric_hash = _hash(rubric_payload)
        acceptance_policy_version = _required(
            "acceptance_policy_version", acceptance_policy_version
        )
        revision = AcceptedGroundTruthRevision(
            ground_truth_revision_id=_id(
                "aeg",
                case.case_id,
                rubric_hash,
                *assessment_ids,
                acceptance_policy_version,
                _hash_text(note),
            ),
            case_id=case.case_id,
            rubric=rubric_payload,
            rubric_hash=rubric_hash,
            accepted_by=_required("accepted_by", accepted_by),
            accepted_at=accepted_at or _now(),
            supporting_assessment_ids=assessment_ids,
            acceptance_policy_version=acceptance_policy_version,
            adjudication_note=note,
        )
        existing = await self._store.get_accepted_ground_truth_revision(
            revision.ground_truth_revision_id
        )
        if existing is not None:
            return existing
        await self._store.record_accepted_ground_truth_revision(revision)
        return revision

    async def accept_ground_truth(
        self,
        *,
        case_id: str,
        rubric: Mapping[str, object],
        accepted_by: str,
        accepted_at: str | None = None,
    ) -> AcceptedGroundTruthRevision:
        case = await self._require_case(case_id)
        await self._store.authorize_agent_evaluation_source(case.source_id, accepted_by)
        _validate_rubric(case.case_kind, rubric)
        rubric_hash = _hash(rubric)
        revision = AcceptedGroundTruthRevision(
            ground_truth_revision_id=_id("aeg", case.case_id, rubric_hash),
            case_id=case.case_id,
            rubric=_canonical_mapping(rubric),
            rubric_hash=rubric_hash,
            accepted_by=_required("accepted_by", accepted_by),
            accepted_at=accepted_at or _now(),
        )
        existing = await self._store.get_accepted_ground_truth_revision(
            revision.ground_truth_revision_id
        )
        if existing is not None:
            return existing
        await self._store.record_accepted_ground_truth_revision(revision)
        return revision

    async def freeze_cohort(
        self,
        *,
        items: Sequence[AgentEvaluationCohortItem],
        selection_policy_version: str,
        created_by: str,
        created_at: str | None = None,
    ) -> AgentEvaluationCohort:
        if not items:
            raise ValueError("cohort requires at least one item")
        ordered = tuple(sorted(items, key=lambda item: (item.case_id, item.ground_truth_revision_id)))
        if len({item.case_id for item in ordered}) != len(ordered):
            raise ValueError("cohort may contain each case exactly once")
        roles_by_group: dict[str, AgentEvaluationRole] = {}
        for item in ordered:
            previous_role = roles_by_group.setdefault(item.group_key, item.role)
            if previous_role is not item.role:
                raise ValueError("one operation family cannot cross cohort roles")
        for item in ordered:
            revision = await self._store.get_accepted_ground_truth_revision(
                item.ground_truth_revision_id
            )
            if revision is None or revision.case_id != item.case_id:
                raise ValueError("cohort item must reference accepted ground truth for its case")
            case = await self._require_case(item.case_id)
            await self._store.authorize_agent_evaluation_source(case.source_id, created_by)
        payload = {
            "items": [_cohort_item_payload(item) for item in ordered],
            "selection_policy_version": selection_policy_version,
        }
        manifest_hash = _hash(payload)
        cohort = AgentEvaluationCohort(
            cohort_id=_id("aeo", OFFLINE_EVALUATION_SCHEMA_VERSION, manifest_hash),
            items=ordered,
            selection_policy_version=_required(
                "selection_policy_version", selection_policy_version
            ),
            manifest_hash=manifest_hash,
            created_by=_required("created_by", created_by),
            created_at=created_at or _now(),
        )
        existing = await self._store.get_agent_evaluation_cohort(cohort.cohort_id)
        if existing is not None:
            return existing
        await self._store.record_agent_evaluation_cohort(cohort)
        return cohort

    async def execute_run(
        self,
        *,
        cohort_id: str,
        candidate_manifest: Mapping[str, object],
        evaluator_suite: str,
        evaluator_version: str,
        created_by: str,
        replicate_count: int = 1,
        baseline_run_id: str | None = None,
        created_at: str | None = None,
    ) -> AgentEvaluationRunReport:
        if replicate_count < 1:
            raise ValueError("replicate_count must be positive")
        _validate_candidate_manifest(candidate_manifest)
        cohort = await self._store.get_agent_evaluation_cohort(cohort_id)
        if cohort is None:
            raise KeyError(f"unknown evaluation cohort: {cohort_id}")
        candidate = _canonical_mapping(candidate_manifest)
        candidate_hash = _hash(candidate)
        run_id = _id(
            "aer",
            cohort.manifest_hash,
            candidate_hash,
            evaluator_suite,
            evaluator_version,
            replicate_count,
            baseline_run_id or "",
        )
        existing = await self._store.get_agent_evaluation_run(run_id)
        if existing is not None and existing.status is not AgentEvaluationRunStatus.RUNNING:
            return await self.read_report(run_id, requesting_user_id=created_by)
        run = AgentEvaluationRun(
            run_id=run_id,
            cohort_id=cohort.cohort_id,
            candidate_manifest=candidate,
            candidate_manifest_hash=candidate_hash,
            evaluator_suite=_required("evaluator_suite", evaluator_suite),
            evaluator_version=_required("evaluator_version", evaluator_version),
            replicate_count=replicate_count,
            status=AgentEvaluationRunStatus.RUNNING,
            created_by=_required("created_by", created_by),
            created_at=(existing.created_at if existing is not None else created_at or _now()),
            baseline_run_id=baseline_run_id,
        )
        if existing is not None:
            run = replace(run, created_by=existing.created_by)
        await self._store.record_agent_evaluation_run(run)
        any_error = False
        for item in cohort.items:
            case = await self._require_case(item.case_id)
            await self._store.authorize_agent_evaluation_source(case.source_id, created_by)
            executor = self._executors.get(case.case_kind)
            if executor is None:
                raise ValueError(f"no offline executor registered for {case.case_kind.value}")
            ground_truth = await self._store.get_accepted_ground_truth_revision(
                item.ground_truth_revision_id
            )
            if ground_truth is None:
                raise RuntimeError("frozen cohort ground truth is unavailable")
            for replicate_ordinal in range(replicate_count):
                candidate_output_key = _id(
                    "aek",
                    case.case_id,
                    candidate_hash,
                    replicate_ordinal,
                )
                result_id = _id("aeres", run_id, case.case_id, replicate_ordinal)
                cached = await self._store.get_agent_evaluation_result(result_id)
                if cached is not None:
                    any_error = any_error or cached.status is not AgentEvaluationResultStatus.COMPLETED
                    continue
                reusable = await self._store.get_cached_agent_evaluation_output(
                    candidate_output_key
                )
                started = perf_counter()
                try:
                    output = (
                        _canonical_mapping(reusable.output)
                        if reusable is not None and reusable.output is not None
                        else _canonical_mapping(await executor.execute(case, candidate))
                    )
                    checks = deterministic_checks(case, ground_truth, output)
                    result = AgentEvaluationResult(
                        result_id=result_id,
                        run_id=run_id,
                        case_id=case.case_id,
                        ground_truth_revision_id=ground_truth.ground_truth_revision_id,
                        replicate_ordinal=replicate_ordinal,
                        candidate_output_key=candidate_output_key,
                        status=AgentEvaluationResultStatus.COMPLETED,
                        output=output,
                        output_hash=_hash(output),
                        duration_ms=max(0, round((perf_counter() - started) * 1000)),
                        created_at=_now(),
                        reused_from_result_id=(
                            reusable.result_id if reusable is not None else None
                        ),
                    )
                except Exception as exc:  # evaluator boundary records unknown/error
                    any_error = True
                    result = AgentEvaluationResult(
                        result_id=result_id,
                        run_id=run_id,
                        case_id=case.case_id,
                        ground_truth_revision_id=ground_truth.ground_truth_revision_id,
                        replicate_ordinal=replicate_ordinal,
                        candidate_output_key=candidate_output_key,
                        status=AgentEvaluationResultStatus.ERROR,
                        output=None,
                        output_hash=None,
                        duration_ms=max(0, round((perf_counter() - started) * 1000)),
                        created_at=_now(),
                        error_code=type(exc).__name__,
                    )
                    checks = (
                        DeterministicCheck(
                            criterion="candidate_execution",
                            label=DeterministicCheckLabel.UNKNOWN,
                            reason_code="candidate_execution_error",
                        ),
                    )
                assessments = tuple(
                    _assessment_for_check(result, check, evaluator_version)
                    for check in checks
                )
                await self._store.record_agent_evaluation_result(result, assessments)
        completed = replace(
            run,
            status=(
                AgentEvaluationRunStatus.FAILED if any_error else AgentEvaluationRunStatus.COMPLETED
            ),
            completed_at=_now(),
        )
        await self._store.record_agent_evaluation_run(completed)
        return await self.read_report(completed.run_id, requesting_user_id=created_by)

    async def read_report(
        self,
        run_id: str,
        *,
        requesting_user_id: str,
    ) -> AgentEvaluationRunReport:
        run = await self._store.get_agent_evaluation_run(run_id)
        if run is None:
            raise KeyError(f"unknown evaluation run: {run_id}")
        cohort = await self._store.get_agent_evaluation_cohort(run.cohort_id)
        if cohort is None:
            raise RuntimeError("evaluation run cohort is unavailable")
        for item in cohort.items:
            case = await self._require_case(item.case_id)
            await self._store.authorize_agent_evaluation_source(
                case.source_id,
                requesting_user_id,
            )
        results = tuple(await self._store.list_agent_evaluation_results(run_id))
        assessments = tuple(await self._store.list_agent_assessments_for_run(run_id))
        counts = {label.value: 0 for label in DeterministicCheckLabel}
        for assessment in assessments:
            if assessment.annotator_kind != "code":
                continue
            key = (
                DeterministicCheckLabel.UNKNOWN.value
                if assessment.label in {None, "needs_review"}
                else assessment.label
            )
            counts[key] += 1
        item_by_case_id = {item.case_id: item for item in cohort.items}
        result_by_id = {result.result_id: result for result in results}
        population_summaries = {
            population.value: {
                "completed": 0,
                "error": 0,
                "artifact_unavailable": 0,
                "pass": 0,
                "fail": 0,
                "unknown": 0,
            }
            for population in AgentEvaluationPopulation
        }
        for result in results:
            population = item_by_case_id[result.case_id].population.value
            population_summaries[population][result.status.value] += 1
        for assessment in assessments:
            if assessment.annotator_kind != "code":
                continue
            result = result_by_id.get(assessment.target_result_id or "")
            if result is None:
                continue
            population = item_by_case_id[result.case_id].population.value
            label = (
                DeterministicCheckLabel.UNKNOWN.value
                if assessment.label in {None, "needs_review"}
                else assessment.label
            )
            if label in population_summaries[population]:
                population_summaries[population][label] += 1
        return AgentEvaluationRunReport(
            run=run,
            results=results,
            assessments=assessments,
            completed_result_count=sum(
                result.status is AgentEvaluationResultStatus.COMPLETED for result in results
            ),
            error_result_count=sum(
                result.status is not AgentEvaluationResultStatus.COMPLETED for result in results
            ),
            check_counts=counts,
            population_summaries=population_summaries,
        )

    async def _human_annotation_context(
        self,
        *,
        result_id: str,
        content_policy_id: str,
        reviewer_id: str,
    ) -> tuple[
        AgentEvaluationResult,
        AgentEvaluationCase,
        AgentEvaluationContentPolicy,
    ]:
        result = await self._store.get_agent_evaluation_result(result_id)
        if result is None:
            raise KeyError(f"unknown evaluation result: {result_id}")
        if result.status is not AgentEvaluationResultStatus.COMPLETED:
            raise ValueError("human annotation requires a completed evaluation result")
        case = await self._require_case(result.case_id)
        await self._store.authorize_agent_evaluation_source(case.source_id, reviewer_id)
        policy = await self._store.get_agent_evaluation_content_policy(content_policy_id)
        if policy is None:
            raise PermissionError("protected evaluation content requires an approved policy")
        if (
            policy.source_id != case.source_id
            or policy.profile is not AgentEvaluationContentProfile.HUMAN_CALIBRATION
        ):
            raise PermissionError("content policy does not authorize this evaluation result")
        run = await self._store.get_agent_evaluation_run(result.run_id)
        if run is None:
            raise RuntimeError("evaluation result run is unavailable")
        cohort = await self._store.get_agent_evaluation_cohort(run.cohort_id)
        if cohort is None:
            raise RuntimeError("evaluation result cohort is unavailable")
        item = next((item for item in cohort.items if item.case_id == case.case_id), None)
        if item is None or item.role is not AgentEvaluationRole.CALIBRATION:
            raise ValueError("human calibration accepts only calibration-role results")
        return result, case, policy

    async def _require_case(self, case_id: str) -> AgentEvaluationCase:
        case = await self._store.get_agent_evaluation_case(case_id)
        if case is None:
            raise KeyError(f"unknown evaluation case: {case_id}")
        return case


def deterministic_checks(
    case: AgentEvaluationCase,
    ground_truth: AcceptedGroundTruthRevision,
    output: Mapping[str, object],
) -> tuple[DeterministicCheck, ...]:
    """Run objective contract checks; semantic quality is intentionally deferred."""

    checks = [
        DeterministicCheck(
            criterion="typed_output",
            label=DeterministicCheckLabel.PASS,
            reason_code="typed_output_valid",
        )
    ]
    if case.case_kind is AgentEvaluationCaseKind.SOURCE_UNIT_DERIVATION:
        extraction = output.get("extraction")
        memories = extraction.get("memories") if isinstance(extraction, Mapping) else None
        if not isinstance(memories, list):
            return (
                DeterministicCheck(
                    criterion="typed_output",
                    label=DeterministicCheckLabel.FAIL,
                    reason_code="derivation_output_invalid",
                ),
            )
        projection = _mapping(case.manifest, "projection")
        revisions = {
            str(item.get("observation_id")): str(item.get("content") or "")
            for item in _mapping_list(projection, "observation_revisions")
        }
        invalid_evidence = sum(
            not _derivation_evidence_resolves(memory, revisions) for memory in memories
        )
        checks.append(
            DeterministicCheck(
                criterion="claim_local_evidence",
                label=(
                    DeterministicCheckLabel.FAIL if invalid_evidence else DeterministicCheckLabel.PASS
                ),
                reason_code=(
                    "claim_local_evidence_missing"
                    if invalid_evidence
                    else "claim_local_evidence_resolved"
                ),
            )
        )
        checks.append(
            DeterministicCheck(
                criterion="derivation_terminal_state",
                label=(
                    DeterministicCheckLabel.FAIL
                    if output.get("error_type") is not None
                    else DeterministicCheckLabel.PASS
                ),
                reason_code=(
                    "derivation_failed"
                    if output.get("error_type") is not None
                    else "derivation_completed"
                ),
            )
        )
    else:
        operations = output.get("operations")
        expected_ids = {
            str(item.get("id")) for item in _mapping_list(case.manifest, "incumbents")
        }
        actual_ids = {
            str(item.get("memory_id"))
            for item in operations
            if isinstance(operations, list)
            and isinstance(item, Mapping)
            and item.get("memory_id") is not None
        } if isinstance(operations, list) else set()
        missing = expected_ids - actual_ids
        checks.append(
            DeterministicCheck(
                criterion="incumbent_coverage",
                label=DeterministicCheckLabel.FAIL if missing else DeterministicCheckLabel.PASS,
                reason_code=(
                    "incumbent_coverage_incomplete" if missing else "incumbent_coverage_complete"
                ),
            )
        )
        checks.append(
            DeterministicCheck(
                criterion="reconciliation_terminal_state",
                label=(
                    DeterministicCheckLabel.FAIL
                    if output.get("failure") is not None
                    else DeterministicCheckLabel.PASS
                ),
                reason_code=(
                    "reconciliation_failed_closed"
                    if output.get("failure") is not None
                    else "reconciliation_completed"
                ),
            )
        )
    required = ground_truth.rubric.get("required_deterministic_criteria", [])
    present = {check.criterion for check in checks}
    if isinstance(required, list):
        for criterion in required:
            if isinstance(criterion, str) and criterion not in present:
                checks.append(
                    DeterministicCheck(
                        criterion=criterion,
                        label=DeterministicCheckLabel.UNKNOWN,
                        reason_code="criterion_not_implemented",
                    )
                )
    return tuple(checks)


def _derivation_evidence_resolves(
    memory: object,
    revisions: Mapping[str, str],
) -> bool:
    if not isinstance(memory, Mapping):
        return False
    observation_id = str(memory.get("source_observation_id") or "")
    quote = str(memory.get("evidence_quote") or "")
    authority = revisions.get(observation_id)
    if authority is None or not quote.strip() or quote not in authority:
        return False
    start = memory.get("evidence_range_start")
    end = memory.get("evidence_range_end")
    if start is None and end is None:
        return True
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(authority)
        and authority[start:end] == quote
    )


def _assessment_for_check(
    result: AgentEvaluationResult,
    check: DeterministicCheck,
    evaluator_version: str,
) -> AgentAssessment:
    label = "needs_review" if check.label is DeterministicCheckLabel.UNKNOWN else check.label.value
    identity = _id(
        "aas",
        result.result_id,
        check.criterion,
        OFFLINE_DETERMINISTIC_EVALUATOR_VERSION,
        evaluator_version,
    )
    return AgentAssessment(
        assessment_id=identity,
        target_event_id=None,
        target_result_id=result.result_id,
        criterion=check.criterion,
        status="completed",
        label=label,
        reason_code=check.reason_code,
        annotator_kind="code",
        evaluator_name="memforge.deterministic.offline_contract",
        evaluator_version=OFFLINE_DETERMINISTIC_EVALUATOR_VERSION,
        created_at=datetime.fromisoformat(result.created_at),
    )


def _validate_case_manifest(
    case_kind: AgentEvaluationCaseKind,
    manifest: Mapping[str, object],
) -> None:
    if case_kind is AgentEvaluationCaseKind.SOURCE_UNIT_DERIVATION:
        _mapping(manifest, "projection")
        _mapping(manifest, "context")
    else:
        _mapping_list(manifest, "new_extractions")
        incumbents = _mapping_list(manifest, "incumbents")
        if not incumbents:
            raise ValueError("reconciliation case requires pinned incumbents")


def _validate_candidate_manifest(manifest: Mapping[str, object]) -> None:
    required = (
        "code_revision",
        "prompt_hash",
        "schema_version",
        "contract_version",
        "model",
        "replay_harness_version",
    )
    missing = [name for name in required if not str(manifest.get(name) or "").strip()]
    if missing:
        raise ValueError("candidate manifest is missing pinned fields: " + ", ".join(missing))


def _validate_rubric(
    case_kind: AgentEvaluationCaseKind,
    rubric: Mapping[str, object],
) -> None:
    if not rubric:
        raise ValueError("accepted ground truth rubric cannot be empty")
    if "required_claims" in rubric and not isinstance(rubric["required_claims"], list):
        raise ValueError("required_claims must be a list")
    if case_kind is AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION:
        forbidden = rubric.get("forbidden_destructive_memory_ids", [])
        if not isinstance(forbidden, list):
            raise ValueError("forbidden_destructive_memory_ids must be a list")


def _raw_memory_from_payload(payload: Mapping[str, object]) -> RawMemory:
    return RawMemory(
        content=str(payload.get("content") or ""),
        memory_type=str(payload.get("memory_type") or "fact"),
        confidence=float(payload.get("confidence") or 0.7),
        entity_refs=[str(value) for value in payload.get("entity_refs", [])],
        valid_from=_optional_str(payload.get("valid_from")),
        valid_until=_optional_str(payload.get("valid_until")),
        extraction_context=_optional_str(payload.get("extraction_context")),
        evidence_quote=_optional_str(payload.get("evidence_quote")),
        evidence_range_start=_optional_int(payload.get("evidence_range_start")),
        evidence_range_end=_optional_int(payload.get("evidence_range_end")),
        evidence_anchor=_optional_str(payload.get("evidence_anchor")),
        source_observation_id=_optional_str(payload.get("source_observation_id")),
        required_source_observation_ids=[
            str(value) for value in payload.get("required_source_observation_ids", [])
        ],
    )


def _memory_from_payload(payload: Mapping[str, object]) -> Memory:
    content = str(payload.get("content") or "")
    return Memory(
        id=str(payload.get("id") or ""),
        memory_type=str(payload.get("memory_type") or "fact"),
        content=content,
        content_hash=str(payload.get("content_hash") or _hash_text(content)),
        visibility=str(payload.get("visibility") or "workspace"),
        owner_user_id=_optional_str(payload.get("owner_user_id")),
        project_key=_optional_str(payload.get("project_key")),
        repo_identifier=_optional_str(payload.get("repo_identifier")),
        entity_refs=[str(value) for value in payload.get("entity_refs", [])],
        confidence=float(payload.get("confidence") or 0.7),
        extraction_context=_optional_str(payload.get("extraction_context")),
        status=str(payload.get("status") or "active"),
    )


def _reconcile_operation_payload(operation: ReconcileOperation) -> dict[str, object]:
    return {
        "action": operation.action.value,
        "memory_id": operation.memory_id,
        "memory": (
            memory_extraction_output_payload(MemoryExtractionResult(memories=[operation.memory]))[
                "memories"
            ][0]
            if operation.memory is not None
            else None
        ),
        "reason_code": _bounded_reason(operation.reason),
        "flag_for_review": operation.flag_for_review,
    }


def _cohort_item_payload(item: AgentEvaluationCohortItem) -> dict[str, object]:
    return {
        "case_id": item.case_id,
        "ground_truth_revision_id": item.ground_truth_revision_id,
        "population": item.population.value,
        "role": item.role.value,
        "group_key": item.group_key,
        "weight": item.weight,
    }


def agent_evaluation_case_to_payload(case: AgentEvaluationCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "case_kind": case.case_kind.value,
        "source_id": case.source_id,
        "doc_id": case.doc_id,
        "source_unit_id": case.source_unit_id,
        "manifest": dict(case.manifest),
        "manifest_hash": case.manifest_hash,
        "promotion_policy_version": case.promotion_policy_version,
        "created_by": case.created_by,
        "created_at": case.created_at,
        "source_runtime_event_id": case.source_runtime_event_id,
        "supersedes_case_id": case.supersedes_case_id,
        "schema_version": case.schema_version,
    }


def agent_evaluation_case_from_payload(payload: Mapping[str, object]) -> AgentEvaluationCase:
    return AgentEvaluationCase(
        case_id=str(payload["case_id"]),
        case_kind=AgentEvaluationCaseKind(str(payload["case_kind"])),
        source_id=str(payload["source_id"]),
        doc_id=str(payload["doc_id"]),
        source_unit_id=str(payload["source_unit_id"]),
        manifest=_mapping(payload, "manifest"),
        manifest_hash=str(payload["manifest_hash"]),
        promotion_policy_version=str(payload["promotion_policy_version"]),
        created_by=str(payload["created_by"]),
        created_at=str(payload["created_at"]),
        source_runtime_event_id=_optional_str(payload.get("source_runtime_event_id")),
        supersedes_case_id=_optional_str(payload.get("supersedes_case_id")),
        schema_version=str(payload.get("schema_version") or OFFLINE_EVALUATION_SCHEMA_VERSION),
    )


def agent_evaluation_content_policy_to_payload(
    policy: AgentEvaluationContentPolicy,
) -> dict[str, object]:
    return {
        "content_policy_id": policy.content_policy_id,
        "source_id": policy.source_id,
        "profile": policy.profile.value,
        "policy_version": policy.policy_version,
        "approved_by": policy.approved_by,
        "approved_at": policy.approved_at,
        "schema_version": policy.schema_version,
    }


def agent_evaluation_content_policy_from_payload(
    payload: Mapping[str, object],
) -> AgentEvaluationContentPolicy:
    return AgentEvaluationContentPolicy(
        content_policy_id=str(payload["content_policy_id"]),
        source_id=str(payload["source_id"]),
        profile=AgentEvaluationContentProfile(str(payload["profile"])),
        policy_version=str(payload["policy_version"]),
        approved_by=str(payload["approved_by"]),
        approved_at=str(payload["approved_at"]),
        schema_version=str(
            payload.get("schema_version") or OFFLINE_CONTENT_POLICY_SCHEMA_VERSION
        ),
    )


def accepted_ground_truth_to_payload(
    revision: AcceptedGroundTruthRevision,
) -> dict[str, object]:
    return {
        "ground_truth_revision_id": revision.ground_truth_revision_id,
        "case_id": revision.case_id,
        "rubric": dict(revision.rubric),
        "rubric_hash": revision.rubric_hash,
        "accepted_by": revision.accepted_by,
        "accepted_at": revision.accepted_at,
        "supporting_assessment_ids": list(revision.supporting_assessment_ids),
        "acceptance_policy_version": revision.acceptance_policy_version,
        "adjudication_note": revision.adjudication_note,
        "schema_version": revision.schema_version,
    }


def accepted_ground_truth_from_payload(
    payload: Mapping[str, object],
) -> AcceptedGroundTruthRevision:
    return AcceptedGroundTruthRevision(
        ground_truth_revision_id=str(payload["ground_truth_revision_id"]),
        case_id=str(payload["case_id"]),
        rubric=_mapping(payload, "rubric"),
        rubric_hash=str(payload["rubric_hash"]),
        accepted_by=str(payload["accepted_by"]),
        accepted_at=str(payload["accepted_at"]),
        supporting_assessment_ids=tuple(
            str(value) for value in payload.get("supporting_assessment_ids", [])
        ),
        acceptance_policy_version=_optional_str(payload.get("acceptance_policy_version")),
        adjudication_note=_optional_str(payload.get("adjudication_note")),
        schema_version=str(payload.get("schema_version") or OFFLINE_EVALUATION_SCHEMA_VERSION),
    )


def agent_evaluation_cohort_to_payload(
    cohort: AgentEvaluationCohort,
) -> dict[str, object]:
    return {
        "cohort_id": cohort.cohort_id,
        "items": [_cohort_item_payload(item) for item in cohort.items],
        "selection_policy_version": cohort.selection_policy_version,
        "manifest_hash": cohort.manifest_hash,
        "created_by": cohort.created_by,
        "created_at": cohort.created_at,
        "schema_version": cohort.schema_version,
    }


def agent_evaluation_cohort_from_payload(
    payload: Mapping[str, object],
) -> AgentEvaluationCohort:
    return AgentEvaluationCohort(
        cohort_id=str(payload["cohort_id"]),
        items=tuple(
            AgentEvaluationCohortItem(
                case_id=str(item["case_id"]),
                ground_truth_revision_id=str(item["ground_truth_revision_id"]),
                population=AgentEvaluationPopulation(str(item["population"])),
                role=AgentEvaluationRole(str(item["role"])),
                group_key=str(item["group_key"]),
                weight=float(item.get("weight") or 1.0),
            )
            for item in _mapping_list(payload, "items")
        ),
        selection_policy_version=str(payload["selection_policy_version"]),
        manifest_hash=str(payload["manifest_hash"]),
        created_by=str(payload["created_by"]),
        created_at=str(payload["created_at"]),
        schema_version=str(payload.get("schema_version") or OFFLINE_EVALUATION_SCHEMA_VERSION),
    )


def agent_evaluation_run_to_payload(run: AgentEvaluationRun) -> dict[str, object]:
    return {
        "run_id": run.run_id,
        "cohort_id": run.cohort_id,
        "candidate_manifest": dict(run.candidate_manifest),
        "candidate_manifest_hash": run.candidate_manifest_hash,
        "evaluator_suite": run.evaluator_suite,
        "evaluator_version": run.evaluator_version,
        "replicate_count": run.replicate_count,
        "status": run.status.value,
        "created_by": run.created_by,
        "created_at": run.created_at,
        "completed_at": run.completed_at,
        "baseline_run_id": run.baseline_run_id,
        "schema_version": run.schema_version,
    }


def agent_evaluation_run_from_payload(payload: Mapping[str, object]) -> AgentEvaluationRun:
    return AgentEvaluationRun(
        run_id=str(payload["run_id"]),
        cohort_id=str(payload["cohort_id"]),
        candidate_manifest=_mapping(payload, "candidate_manifest"),
        candidate_manifest_hash=str(payload["candidate_manifest_hash"]),
        evaluator_suite=str(payload["evaluator_suite"]),
        evaluator_version=str(payload["evaluator_version"]),
        replicate_count=int(payload["replicate_count"]),
        status=AgentEvaluationRunStatus(str(payload["status"])),
        created_by=str(payload["created_by"]),
        created_at=str(payload["created_at"]),
        completed_at=_optional_str(payload.get("completed_at")),
        baseline_run_id=_optional_str(payload.get("baseline_run_id")),
        schema_version=str(payload.get("schema_version") or OFFLINE_EVALUATION_SCHEMA_VERSION),
    )


def agent_evaluation_result_to_payload(
    result: AgentEvaluationResult,
) -> dict[str, object]:
    return {
        "result_id": result.result_id,
        "run_id": result.run_id,
        "case_id": result.case_id,
        "ground_truth_revision_id": result.ground_truth_revision_id,
        "replicate_ordinal": result.replicate_ordinal,
        "candidate_output_key": result.candidate_output_key,
        "status": result.status.value,
        "output": dict(result.output) if result.output is not None else None,
        "output_hash": result.output_hash,
        "duration_ms": result.duration_ms,
        "created_at": result.created_at,
        "error_code": result.error_code,
        "reused_from_result_id": result.reused_from_result_id,
        "schema_version": result.schema_version,
    }


def agent_evaluation_result_from_payload(
    payload: Mapping[str, object],
) -> AgentEvaluationResult:
    output = payload.get("output")
    if output is not None and not isinstance(output, Mapping):
        raise ValueError("evaluation result output must be an object")
    return AgentEvaluationResult(
        result_id=str(payload["result_id"]),
        run_id=str(payload["run_id"]),
        case_id=str(payload["case_id"]),
        ground_truth_revision_id=str(payload["ground_truth_revision_id"]),
        replicate_ordinal=int(payload["replicate_ordinal"]),
        candidate_output_key=str(payload["candidate_output_key"]),
        status=AgentEvaluationResultStatus(str(payload["status"])),
        output=output,
        output_hash=_optional_str(payload.get("output_hash")),
        duration_ms=int(payload["duration_ms"]),
        created_at=str(payload["created_at"]),
        error_code=_optional_str(payload.get("error_code")),
        reused_from_result_id=_optional_str(payload.get("reused_from_result_id")),
        schema_version=str(payload.get("schema_version") or OFFLINE_EVALUATION_SCHEMA_VERSION),
    )


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _mapping_list(payload: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of objects")
    return list(value)


def _positive_int(value: object, *, default: int) -> int:
    return value if isinstance(value, int) and value > 0 else default


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _bounded_reason(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _required(name: str, value: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return json.loads(_canonical_json(value))


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:24]}"
