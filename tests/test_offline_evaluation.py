from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.evals.agent_evaluation import AgentAssessment
from memforge.evals.offline_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationCaseKind,
    AgentEvaluationCohortItem,
    AgentEvaluationPopulation,
    AgentEvaluationRole,
    OfflineAgentEvaluation,
    SourceUnitDerivationReplayExecutor,
)
from memforge.models import (
    ContentItem,
    DocumentRecord,
    MemoryExtractionResult,
    NormalizedContent,
    RawContent,
    RawMemory,
)
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_derivation import (
    SourceUnitDerivationContext,
    source_unit_derivation_context_to_payload,
)
from memforge.source_projection import source_projection_to_payload
from memforge.storage.database import Database


class _FixedExecutor:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, case, candidate_manifest):
        self.calls += 1
        del candidate_manifest
        incumbent_id = str(case.manifest["incumbents"][0]["id"])
        return {
            "case_kind": case.case_kind.value,
            "operations": [
                {
                    "action": "noop",
                    "memory_id": incumbent_id,
                    "memory": None,
                    "reason_code": "bounded",
                    "flag_for_review": False,
                }
            ],
            "failure": None,
        }


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "offline-evaluation.db"))
    await database.connect()
    await database.upsert_source(
        id="src-teams",
        type="teams",
        name="Teams",
        config_json="{}",
        owner_user_id="owner-1",
        access_policy="workspace",
    )
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_offline_evaluation_records_frozen_lineage_and_result_assessments(db) -> None:
    executor = _FixedExecutor()
    evaluation = OfflineAgentEvaluation(
        db,
        executors={AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION: executor},
    )
    case = await evaluation.curate_case(
        case_kind=AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION,
        source_id="src-teams",
        doc_id="doc-channel",
        source_unit_id="teams-channel:42",
        manifest={
            "new_extractions": [],
            "incumbents": [
                {
                    "id": "mem-1",
                    "content": "Tracing starts with traceId.",
                    "content_hash": "a" * 64,
                    "memory_type": "procedure",
                }
            ],
            "doc_type": "teams",
            "updated_document": "Tracing starts with traceId.",
        },
        promotion_policy_version="manual-v1",
        created_by="reviewer-1",
        created_at="2026-08-17T10:00:00+00:00",
    )
    repeated = await evaluation.curate_case(
        case_kind=case.case_kind,
        source_id=case.source_id,
        doc_id=case.doc_id,
        source_unit_id=case.source_unit_id,
        manifest=case.manifest,
        promotion_policy_version=case.promotion_policy_version,
        created_by=case.created_by,
        created_at=case.created_at,
    )
    assert repeated.case_id == case.case_id

    ground_truth = await evaluation.accept_ground_truth(
        case_id=case.case_id,
        rubric={"required_deterministic_criteria": ["incumbent_coverage"]},
        accepted_by="reviewer-2",
        accepted_at="2026-08-17T10:01:00+00:00",
    )
    cohort = await evaluation.freeze_cohort(
        items=(
            AgentEvaluationCohortItem(
                case_id=case.case_id,
                ground_truth_revision_id=ground_truth.ground_truth_revision_id,
                population=AgentEvaluationPopulation.FAILURE_REGRESSION,
                role=AgentEvaluationRole.SENTINEL,
                group_key="teams-channel:42",
            ),
        ),
        selection_policy_version="manual-sentinel-v1",
        created_by="reviewer-2",
        created_at="2026-08-17T10:02:00+00:00",
    )
    report = await evaluation.execute_run(
        cohort_id=cohort.cohort_id,
        candidate_manifest={
            "code_revision": "abc123",
            "prompt_hash": "a" * 64,
            "schema_version": "schema-v1",
            "contract_version": "reconciliation-v1",
            "model": "fixed",
            "replay_harness_version": "1",
        },
        evaluator_suite="deterministic-contracts",
        evaluator_version="1",
        created_by="ci",
        created_at="2026-08-17T10:03:00+00:00",
    )

    assert report.run.status.value == "completed"
    assert report.completed_result_count == 1
    assert report.error_result_count == 0
    assert report.check_counts == {"pass": 3, "fail": 0, "unknown": 0}
    assert report.population_summaries == {
        "failure_regression": {
            "completed": 1,
            "error": 0,
            "artifact_unavailable": 0,
            "pass": 3,
            "fail": 0,
            "unknown": 0,
        },
        "representative_control": {
            "completed": 0,
            "error": 0,
            "artifact_unavailable": 0,
            "pass": 0,
            "fail": 0,
            "unknown": 0,
        },
    }
    assert {assessment.target_result_id for assessment in report.assessments} == {
        report.results[0].result_id
    }
    assert {assessment.criterion for assessment in report.assessments} == {
        "typed_output",
        "incumbent_coverage",
        "reconciliation_terminal_state",
    }
    content_policy = await evaluation.approve_human_calibration_content(
        source_id=case.source_id,
        policy_version="workspace-human-review-v1",
        approved_by="reviewer-2",
    )
    with pytest.raises(ValueError, match="calibration-role"):
        await evaluation.prepare_human_annotation(
            result_id=report.results[0].result_id,
            content_policy_id=content_policy.content_policy_id,
            reviewer_id="reviewer-2",
        )

    replayed = await evaluation.execute_run(
        cohort_id=cohort.cohort_id,
        candidate_manifest={
            "code_revision": "abc123",
            "prompt_hash": "a" * 64,
            "schema_version": "schema-v1",
            "contract_version": "reconciliation-v1",
            "model": "fixed",
            "replay_harness_version": "1",
        },
        evaluator_suite="deterministic-contracts",
        evaluator_version="1",
        created_by="ci",
    )
    assert replayed == report
    rescored = await evaluation.execute_run(
        cohort_id=cohort.cohort_id,
        candidate_manifest={
            "code_revision": "abc123",
            "prompt_hash": "a" * 64,
            "schema_version": "schema-v1",
            "contract_version": "reconciliation-v1",
            "model": "fixed",
            "replay_harness_version": "1",
        },
        evaluator_suite="deterministic-contracts",
        evaluator_version="2",
        created_by="ci",
    )
    assert rescored.run.run_id != report.run.run_id
    assert rescored.results[0].reused_from_result_id == report.results[0].result_id
    assert executor.calls == 1

    await db.db.execute("DELETE FROM sources WHERE id = ?", (case.source_id,))
    await db.db.commit()
    case_count = await db.db.execute_fetchall("SELECT COUNT(*) AS total FROM agent_evaluation_cases")
    result_count = await db.db.execute_fetchall("SELECT COUNT(*) AS total FROM agent_evaluation_results")
    assessment_count = await db.db.execute_fetchall("SELECT COUNT(*) AS total FROM agent_assessments")
    assert case_count[0]["total"] == 0
    assert result_count[0]["total"] == 0
    assert assessment_count[0]["total"] == 0


@pytest.mark.asyncio
async def test_human_calibration_is_policy_gated_and_adjudication_preserves_labels(db) -> None:
    executor = _FixedExecutor()
    evaluation = OfflineAgentEvaluation(
        db,
        executors={AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION: executor},
    )
    case = await evaluation.curate_case(
        case_kind=AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION,
        source_id="src-teams",
        doc_id="doc-calibration",
        source_unit_id="teams-channel:calibration",
        manifest={
            "new_extractions": [],
            "incumbents": [
                {
                    "id": "mem-calibration",
                    "content": "Tracing starts with traceId.",
                    "memory_type": "procedure",
                }
            ],
            "updated_document": "Tracing starts with traceId.",
        },
        promotion_policy_version="manual-v1",
        created_by="reviewer-1",
    )
    seed_reference = await evaluation.accept_ground_truth(
        case_id=case.case_id,
        rubric={"required_deterministic_criteria": ["incumbent_coverage"]},
        accepted_by="reviewer-1",
    )
    cohort = await evaluation.freeze_cohort(
        items=(
            AgentEvaluationCohortItem(
                case_id=case.case_id,
                ground_truth_revision_id=seed_reference.ground_truth_revision_id,
                population=AgentEvaluationPopulation.FAILURE_REGRESSION,
                role=AgentEvaluationRole.CALIBRATION,
                group_key="teams-channel:calibration",
            ),
        ),
        selection_policy_version="calibration-v1",
        created_by="reviewer-1",
    )
    report = await evaluation.execute_run(
        cohort_id=cohort.cohort_id,
        candidate_manifest={
            "code_revision": "abc123",
            "prompt_hash": "a" * 64,
            "schema_version": "schema-v1",
            "contract_version": "reconciliation-v1",
            "model": "fixed",
            "replay_harness_version": "1",
        },
        evaluator_suite="deterministic-contracts",
        evaluator_version="1",
        created_by="reviewer-1",
    )
    result = report.results[0]

    with pytest.raises(PermissionError, match="approved policy"):
        await evaluation.prepare_human_annotation(
            result_id=result.result_id,
            content_policy_id="aep-missing",
            reviewer_id="reviewer-1",
        )

    policy = await evaluation.approve_human_calibration_content(
        source_id=case.source_id,
        policy_version="workspace-human-review-v1",
        approved_by="reviewer-1",
    )
    task = await evaluation.prepare_human_annotation(
        result_id=result.result_id,
        content_policy_id=policy.content_policy_id,
        reviewer_id="reviewer-1",
    )
    assert task.case_manifest["updated_document"] == "Tracing starts with traceId."
    assert task.candidate_output["operations"][0]["memory_id"] == "mem-calibration"
    assert not hasattr(task, "ground_truth")

    first = await evaluation.record_human_annotation(
        result_id=result.result_id,
        content_policy_id=policy.content_policy_id,
        criterion="semantic_intent",
        label="fail",
        reason_code="required_claim_missing",
        rubric_version="semantic-rubric-v1",
        reviewer_id="reviewer-1",
    )
    metadata_report = await evaluation.read_report(
        report.run.run_id,
        requesting_user_id="reviewer-2",
    )
    assert metadata_report.results[0].output is None
    assert metadata_report.results[0].output_hash == result.output_hash
    assert all(item.annotator_kind == "code" for item in metadata_report.assessments)
    assert first.assessment_id not in {
        item.assessment_id for item in metadata_report.assessments
    }
    repeated = await evaluation.record_human_annotation(
        result_id=result.result_id,
        content_policy_id=policy.content_policy_id,
        criterion="semantic_intent",
        label="fail",
        reason_code="required_claim_missing",
        rubric_version="semantic-rubric-v1",
        reviewer_id="reviewer-1",
    )
    assert repeated == first
    second = await evaluation.record_human_annotation(
        result_id=result.result_id,
        content_policy_id=policy.content_policy_id,
        criterion="semantic_intent",
        label="pass",
        reason_code="required_claim_present",
        rubric_version="semantic-rubric-v1",
        reviewer_id="reviewer-2",
    )

    with pytest.raises(ValueError, match="exactly two"):
        await evaluation.adjudicate_ground_truth(
            case_id=case.case_id,
            supporting_assessment_ids=(first.assessment_id, first.assessment_id),
            rubric={"required_claims": ["traceId starts the diagnostic workflow"]},
            acceptance_policy_version="two-reviewer-v1",
            adjudication_note="The required claim is present in the authority.",
            accepted_by="adjudicator-1",
        )

    adjudicated = await evaluation.adjudicate_ground_truth(
        case_id=case.case_id,
        supporting_assessment_ids=(first.assessment_id, second.assessment_id),
        rubric={"required_claims": ["traceId starts the diagnostic workflow"]},
        acceptance_policy_version="two-reviewer-v1",
        adjudication_note="The accepted reference keeps the durable tracing workflow.",
        accepted_by="adjudicator-1",
    )
    assert adjudicated.supporting_assessment_ids == tuple(
        sorted((first.assessment_id, second.assessment_id))
    )
    assert adjudicated.ground_truth_revision_id != seed_reference.ground_truth_revision_id
    preserved = await db.list_agent_assessments_for_result(result.result_id)
    assert {item.assessment_id for item in preserved} >= {
        first.assessment_id,
        second.assessment_id,
    }
    assert {item.annotator_id for item in preserved if item.annotator_kind == "human"} == {
        "reviewer-1",
        "reviewer-2",
    }
    calibrated_report = await evaluation.read_report(
        report.run.run_id,
        requesting_user_id="reviewer-1",
    )
    assert calibrated_report.check_counts == {"pass": 3, "fail": 0, "unknown": 0}


def test_non_human_assessment_rejects_human_review_provenance() -> None:
    with pytest.raises(ValueError, match="human-review provenance"):
        AgentAssessment(
            assessment_id="aas-code-with-policy",
            target_event_id=None,
            target_result_id="aeres-result",
            criterion="typed_output",
            status="completed",
            label="pass",
            reason_code="typed_output_valid",
            annotator_kind="code",
            evaluator_name="memforge.deterministic.offline_contract",
            evaluator_version="1",
            content_policy_id="aep-policy",
            created_at=datetime.now(timezone.utc),
        )


@pytest.mark.asyncio
async def test_derivation_replay_uses_shared_planner_without_durable_staging() -> None:
    body = "Use traceId to correlate the CLS and application logs."
    item = ContentItem(
        item_id="doc-channel",
        title="Channel",
        source_url="https://teams.example.test/channel/42",
        last_modified=datetime(2026, 8, 17, tzinfo=timezone.utc),
        version="1",
        extra={"page_id": "42", "space_key": "ENG"},
    )
    projection = project_source_item(
        source_id="src-teams",
        source_type="confluence",
        run_id="projection-1",
        item=item,
        raw=RawContent(item=item, body=body.encode(), content_type="text/plain"),
        normalized=NormalizedContent(item=item, markdown_body=body),
    )
    context = SourceUnitDerivationContext(
        document=DocumentRecord(
            doc_id=item.item_id,
            source="src-teams",
            source_url=item.source_url,
            title=item.title,
            space_or_project="42",
            author=None,
            last_modified=item.last_modified,
            labels=[],
            version=item.version,
            content_hash="b" * 64,
            token_count=None,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=item.last_modified,
        ),
        doc_type="confluence",
        project_key=None,
        repo_identifier=None,
        document_content=body,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=item.last_modified.isoformat(),
        user_id=None,
        source_activity_epoch=None,
    )
    seen_batches = []

    async def extract(batch, candidate_manifest):
        seen_batches.append((batch.id, candidate_manifest["prompt_hash"]))
        return MemoryExtractionResult(
            memories=[
                RawMemory(
                    content="traceId correlates CLS and application logs.",
                    memory_type="procedure",
                    evidence_quote=body,
                    source_observation_id=batch.primary_observation_ids[0],
                )
            ]
        )

    executor = SourceUnitDerivationReplayExecutor(extract)
    output = await executor.execute(
        AgentEvaluationCase(
            case_id="aec-derivation",
            case_kind=AgentEvaluationCaseKind.SOURCE_UNIT_DERIVATION,
            source_id="src-teams",
            doc_id=item.item_id,
            source_unit_id=projection.source_units[0].id,
            manifest={
                "projection": source_projection_to_payload(projection),
                "context": source_unit_derivation_context_to_payload(context),
            },
            manifest_hash="c" * 64,
            promotion_policy_version="manual-v1",
            created_by="reviewer",
            created_at="2026-08-17T10:00:00+00:00",
        ),
        {"prompt_hash": "d" * 64, "max_concurrent": 1},
    )

    assert seen_batches
    assert output["error_type"] is None
    assert output["extraction"]["memories"][0]["evidence_quote"] == body


@pytest.mark.asyncio
async def test_cohort_rejects_operation_family_leakage_across_roles(db) -> None:
    evaluation = OfflineAgentEvaluation(db, executors={})
    # The grouping invariant is checked independently of model execution.
    with pytest.raises(ValueError, match="operation family"):
        await evaluation.freeze_cohort(
            items=(
                AgentEvaluationCohortItem(
                    case_id="case-a",
                    ground_truth_revision_id="truth-a",
                    population=AgentEvaluationPopulation.FAILURE_REGRESSION,
                    role=AgentEvaluationRole.DEVELOPMENT,
                    group_key="family-1",
                ),
                AgentEvaluationCohortItem(
                    case_id="case-b",
                    ground_truth_revision_id="truth-b",
                    population=AgentEvaluationPopulation.REPRESENTATIVE_CONTROL,
                    role=AgentEvaluationRole.RELEASE_HOLDOUT,
                    group_key="family-1",
                ),
            ),
            selection_policy_version="v1",
            created_by="reviewer",
        )


@pytest.mark.asyncio
async def test_offline_case_curation_enforces_private_source_owner(db) -> None:
    await db.upsert_source(
        id="src-private",
        type="teams",
        name="Private Teams",
        config_json="{}",
        owner_user_id="owner-1",
        access_policy="private",
    )
    evaluation = OfflineAgentEvaluation(db, executors={})
    with pytest.raises(PermissionError, match="requires its owner"):
        await evaluation.curate_case(
            case_kind=AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION,
            source_id="src-private",
            doc_id="doc-private",
            source_unit_id="unit-private",
            manifest={
                "new_extractions": [],
                "incumbents": [
                    {
                        "id": "mem-private",
                        "content": "A private claim.",
                        "memory_type": "fact",
                    }
                ],
            },
            promotion_policy_version="manual-v1",
            created_by="not-owner",
        )
