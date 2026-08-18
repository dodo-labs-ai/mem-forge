from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from memforge.config import AppConfig
from memforge.evals.offline_evaluation import (
    AgentEvaluationCaseKind,
    AgentEvaluationCohortItem,
    AgentEvaluationExecutionState,
    AgentEvaluationPopulation,
    AgentEvaluationRole,
    AgentEvaluationResultStatus,
    AgentEvaluationRunStatus,
    OfflineAgentEvaluation,
    OfflineArtifactUnavailable,
)
from memforge.evals.offline_worker import OfflineEvaluationWorker
from memforge.server.admin_api import create_admin_app
from memforge.storage.database import Database


class _Executor:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, case, candidate_manifest):
        self.calls += 1
        del candidate_manifest
        return {
            "case_kind": case.case_kind.value,
            "operations": [
                {
                    "action": "noop",
                    "memory_id": "mem-1",
                    "memory": None,
                    "reason_code": "unchanged",
                    "flag_for_review": False,
                }
            ],
            "failure": None,
        }


class _UnavailableExecutor:
    async def execute(self, case, candidate_manifest):
        del case, candidate_manifest
        raise OfflineArtifactUnavailable("pinned artifact is unavailable")


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "offline-worker.db"))
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


async def _admit_run(database, executor):
    evaluation = OfflineAgentEvaluation(
        database,
        executors={AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION: executor},
    )
    case = await evaluation.curate_case(
        case_kind=AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION,
        source_id="src-teams",
        doc_id="doc-1",
        source_unit_id="unit-1",
        manifest={
            "new_extractions": [],
            "incumbents": [
                {
                    "id": "mem-1",
                    "content": "Tracing starts with traceId.",
                    "memory_type": "procedure",
                }
            ],
            "updated_document": "Tracing starts with traceId.",
        },
        promotion_policy_version="manual-v1",
        created_by="owner-1",
    )
    truth = await evaluation.accept_ground_truth(
        case_id=case.case_id,
        rubric={"required_deterministic_criteria": ["incumbent_coverage"]},
        accepted_by="owner-1",
    )
    cohort = await evaluation.freeze_cohort(
        items=(
            AgentEvaluationCohortItem(
                case_id=case.case_id,
                ground_truth_revision_id=truth.ground_truth_revision_id,
                population=AgentEvaluationPopulation.FAILURE_REGRESSION,
                role=AgentEvaluationRole.SENTINEL,
                group_key="unit-1",
            ),
        ),
        selection_policy_version="worker-v1",
        created_by="owner-1",
    )
    run, execution = await evaluation.admit_run(
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
        created_by="owner-1",
    )
    assert execution is not None
    return evaluation, run, execution


@pytest.mark.asyncio
async def test_admission_is_durable_idempotent_and_does_not_execute(db) -> None:
    executor = _Executor()
    evaluation, run, execution = await _admit_run(db, executor)

    repeated_run, repeated_execution = await evaluation.admit_run(
        cohort_id=run.cohort_id,
        candidate_manifest=run.candidate_manifest,
        evaluator_suite=run.evaluator_suite,
        evaluator_version=run.evaluator_version,
        created_by=run.created_by,
    )

    assert repeated_run.run_id == run.run_id
    assert repeated_execution == execution
    assert execution.state is AgentEvaluationExecutionState.QUEUED
    assert executor.calls == 0


@pytest.mark.asyncio
async def test_expired_execution_lease_is_recovered_with_a_new_attempt(db) -> None:
    _, run, _ = await _admit_run(db, _Executor())
    now = datetime(2026, 8, 18, 10, 0, tzinfo=UTC)
    first = await db.claim_agent_evaluation_run(
        worker_id="worker-1",
        now=now.isoformat(),
        lease_expires_at=(now + timedelta(seconds=30)).isoformat(),
    )
    assert first is not None
    assert first.run_id == run.run_id
    assert first.attempt_count == 1
    assert await db.claim_agent_evaluation_run(
        worker_id="worker-2",
        now=(now + timedelta(seconds=10)).isoformat(),
        lease_expires_at=(now + timedelta(seconds=40)).isoformat(),
    ) is None

    recovered = await db.claim_agent_evaluation_run(
        worker_id="worker-2",
        now=(now + timedelta(seconds=31)).isoformat(),
        lease_expires_at=(now + timedelta(seconds=61)).isoformat(),
    )
    assert recovered is not None
    assert recovered.worker_id == "worker-2"
    assert recovered.attempt_count == 2
    assert recovered.lease_token != first.lease_token


@pytest.mark.asyncio
async def test_worker_executes_admitted_run_and_closes_its_lease(db) -> None:
    executor = _Executor()
    evaluation, run, _ = await _admit_run(db, executor)

    async def factory(claimed_run):
        assert claimed_run.run_id == run.run_id
        return evaluation

    worker = OfflineEvaluationWorker(
        db,
        evaluation_factory=factory,
        worker_id="worker-1",
        lease_seconds=30,
    )
    claimed = await worker.run_once()

    assert claimed is not None
    stored = await db.get_agent_evaluation_run(run.run_id)
    execution = await db.get_agent_evaluation_run_execution(run.run_id)
    assert stored is not None
    assert stored.status is AgentEvaluationRunStatus.COMPLETED
    assert execution is not None
    assert execution.state is AgentEvaluationExecutionState.COMPLETED
    assert execution.worker_id is None
    assert execution.lease_token is None
    assert executor.calls == 1


@pytest.mark.asyncio
async def test_worker_records_unavailable_pinned_artifact_as_distinct_unknown(db) -> None:
    evaluation, run, _ = await _admit_run(db, _UnavailableExecutor())

    async def factory(_run):
        return evaluation

    await OfflineEvaluationWorker(
        db,
        evaluation_factory=factory,
        worker_id="worker-1",
    ).run_once()

    stored = await db.get_agent_evaluation_run(run.run_id)
    [result] = await db.list_agent_evaluation_results(run.run_id)
    assert stored is not None
    assert stored.status is AgentEvaluationRunStatus.FAILED
    assert result.status is AgentEvaluationResultStatus.ARTIFACT_UNAVAILABLE
    assert result.error_code == "OfflineArtifactUnavailable"


@pytest.mark.asyncio
async def test_run_api_is_idempotent_and_returns_content_free_status(db, tmp_path) -> None:
    executor = _Executor()
    evaluation, run, _ = await _admit_run(db, executor)
    config = AppConfig(base_dir=tmp_path / "api")
    config.sync.worker_enabled = False
    app = create_admin_app(db=db, config=config)
    request = {
        "cohort_id": run.cohort_id,
        "candidate_manifest": dict(run.candidate_manifest),
        "evaluator_suite": run.evaluator_suite,
        "evaluator_version": run.evaluator_version,
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        admitted = await client.post("/api/v1/agent-evaluations/runs", json=request)
        assert admitted.status_code == 202, admitted.text
        assert admitted.json()["execution"]["state"] == "queued"

        async def factory(_run):
            return evaluation

        await OfflineEvaluationWorker(
            db,
            evaluation_factory=factory,
            worker_id="worker-1",
        ).run_once()
        status = await client.get(f"/api/v1/agent-evaluations/runs/{run.run_id}")

    assert status.status_code == 200, status.text
    payload = status.json()
    assert payload["execution"]["state"] == "completed"
    assert payload["summary"]["completed_result_count"] == 1
    assert payload["results"][0]["output"] is None
