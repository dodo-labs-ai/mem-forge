from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace

import httpx
import pytest

from memforge.config import AppConfig
from memforge.evals.offline_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationCaseKind,
    AgentEvaluationPopulation,
    AgentEvaluationRole,
    OfflineAgentEvaluation,
    SourceUnitDerivationReplayExecutor,
    deterministic_checks,
    run_verdict,
)
from memforge.evals.offline_run import render_report_markdown, report_verdict, wait_for_report
from memforge.evals.source_unit_derivation_cases import (
    DerivationCaseRequest,
    seed_source_unit_derivation_cases,
)
from memforge.models import MemoryExtractionResult
from memforge.pipeline.projection_context import CommittedSourceUnitSnapshot, plan_projection_evidence_work
from memforge.release import current_release
from memforge.source_derivation import source_unit_derivation_context_to_payload
from memforge.source_projection import source_projection_to_payload
from memforge.storage.database import Database
from tests.llm_fixture import fixture_request_planner
from tests.test_sync_bookkeeping import ProjectionFragmentRecordingExtractor, _stage_completed_v9_recovery_attempt

SOURCE_ID = "src-quality"
DOC_ID = "doc-v9-recovery"
ACTOR = "dev"
CANDIDATE = {
    "code_revision": "abc123",
    "prompt_hash": "a" * 64,
    "schema_version": "schema-v1",
    "contract_version": "extraction-v1",
    "model": "fixed",
    "replay_harness_version": "1",
}
RUBRIC = {"expectations": ["A memory states that recovery uses the active extraction contract."]}


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "quality.db"))
    await database.connect()
    yield database
    await database.close()


async def _synced_document(db: Database):
    """A Document with its projection and the completed first derivation a sync leaves behind."""

    attempt = await _stage_completed_v9_recovery_attempt(db, source_id=SOURCE_ID)
    await db.upsert_document(attempt.context.document)
    await db.record_source_projection(attempt.projection)
    return attempt


async def test_seeding_pins_the_first_derivation_of_each_named_document_in_one_cohort(db) -> None:
    attempt = await _synced_document(db)
    evaluation = OfflineAgentEvaluation(db, executors={})
    requests = [
        DerivationCaseRequest(doc_id=DOC_ID, rubric=RUBRIC),
        DerivationCaseRequest(doc_id="doc-never-synced", rubric=RUBRIC),
    ]

    report = await seed_source_unit_derivation_cases(
        db, evaluation, requests=requests, role=AgentEvaluationRole.SENTINEL,
        selection_policy_version="quick-v1", actor=ACTOR,
    )
    again = await seed_source_unit_derivation_cases(
        db, evaluation, requests=requests, role=AgentEvaluationRole.SENTINEL,
        selection_policy_version="quick-v1", actor=ACTOR,
    )

    [pinned] = report.cases
    case = await db.get_agent_evaluation_case(pinned.case_id)
    cohort = await db.get_agent_evaluation_cohort(report.cohort_id)
    assert again.cohort_id == report.cohort_id
    assert report.skipped == {
        "document_not_found": ("doc-never-synced",), "title_not_unique": (), "no_first_derivation": (),
        "derivation_not_found": (), "base_unavailable": (),
    }
    assert (case.case_kind, case.doc_id, case.source_unit_id) == (
        AgentEvaluationCaseKind.SOURCE_UNIT_DERIVATION, DOC_ID, attempt.source_unit_id,
    )
    # The case replays exactly the input the derivation read.
    assert case.manifest["projection"] == source_projection_to_payload(attempt.projection)
    assert case.manifest["context"] == source_unit_derivation_context_to_payload(attempt.context)
    assert case.manifest["access_context_hash"] == attempt.authority_plan_identity["access_context_hash"]
    assert case.manifest["inference_capability_hash"] == attempt.authority_plan_identity["inference_capability_hash"]
    [item] = cohort.items
    assert (item.role, item.population, item.ground_truth_revision_id) == (
        AgentEvaluationRole.SENTINEL, AgentEvaluationPopulation.REPRESENTATIVE_CONTROL,
        pinned.ground_truth_revision_id,
    )


async def test_a_document_can_be_named_by_its_title_within_its_source(db) -> None:
    attempt = await _synced_document(db)
    title = attempt.context.document.title
    evaluation = OfflineAgentEvaluation(db, executors={})

    by_title = await seed_source_unit_derivation_cases(
        db, evaluation,
        requests=[DerivationCaseRequest(source_id=SOURCE_ID, title=title, rubric=RUBRIC)],
        role=AgentEvaluationRole.SENTINEL, selection_policy_version="quick-v1", actor=ACTOR,
    )
    await db.upsert_document(replace(attempt.context.document, doc_id="doc-same-title"))
    ambiguous = await seed_source_unit_derivation_cases(
        db, evaluation,
        requests=[DerivationCaseRequest(source_id=SOURCE_ID, title=title, rubric=RUBRIC)],
        role=AgentEvaluationRole.SENTINEL, selection_policy_version="quick-v1", actor=ACTOR,
    )

    assert [case.doc_id for case in by_title.cases] == [DOC_ID]
    assert ambiguous.skipped["title_not_unique"] == (f"{SOURCE_ID}:{title}",)
    with pytest.raises(ValueError, match="doc_id, or source_id and title"):
        DerivationCaseRequest(doc_id=DOC_ID, source_id=SOURCE_ID, title=title, rubric=RUBRIC)


async def test_a_first_derivation_whose_lifecycle_was_applied_is_pinned(db) -> None:
    attempt = await _synced_document(db)
    await db.db.execute("UPDATE source_derivation_attempts SET status = 'applied' WHERE id = ?", (attempt.id,))
    await db.db.commit()

    report = await seed_source_unit_derivation_cases(
        db, OfflineAgentEvaluation(db, executors={}),
        requests=[DerivationCaseRequest(doc_id=DOC_ID, rubric=RUBRIC)],
        role=AgentEvaluationRole.SENTINEL, selection_policy_version="quick-v1", actor=ACTOR,
    )

    assert [case.doc_id for case in report.cases] == [DOC_ID]


async def test_a_document_derived_only_against_a_committed_base_is_not_pinned(db) -> None:
    attempt = await _synced_document(db)

    class OnlyIncrementalDerivations:
        get_document = db.get_document
        find_source_unit_by_document_id = db.find_source_unit_by_document_id

        async def list_source_derivation_attempts(self, **_filters):
            return (replace(attempt, base_unit_revision_id="unitrev-base"),)

    report = await seed_source_unit_derivation_cases(
        OnlyIncrementalDerivations(), OfflineAgentEvaluation(db, executors={}),
        requests=[DerivationCaseRequest(doc_id=DOC_ID, rubric=RUBRIC)],
        role=AgentEvaluationRole.SENTINEL, selection_policy_version="quick-v1", actor=ACTOR,
    )

    assert (report.cohort_id, report.cases) == (None, ())
    assert report.skipped["no_first_derivation"] == (DOC_ID,)


UPDATED_BODY = (
    "# Current rule\n\nUse the active extraction contract for recovery.\n\nRetry a failed batch once.\n"
)


async def _updated_document(db: Database):
    """The Document after a later sync: its first derivation and the derivation of the update."""

    first = await _synced_document(db)
    update = await _stage_completed_v9_recovery_attempt(db, source_id=SOURCE_ID, body=UPDATED_BODY, base=first)
    await db.record_source_projection(update.projection)
    return first, update


async def _pin(db: Database, request: DerivationCaseRequest, store=None):
    return await seed_source_unit_derivation_cases(
        store or db, OfflineAgentEvaluation(db, executors={}), requests=[request],
        role=AgentEvaluationRole.SENTINEL, selection_policy_version="quick-v1", actor=ACTOR,
    )


async def test_a_named_update_is_pinned_with_the_base_revision_it_was_derived_against(db) -> None:
    first, update = await _updated_document(db)
    assert update.base_unit_revision_id == first.projection.source_unit_revisions[0].id

    report = await _pin(db, DerivationCaseRequest(doc_id=DOC_ID, derivation_id=update.id, rubric=RUBRIC))
    by_default = await _pin(db, DerivationCaseRequest(doc_id=DOC_ID, rubric=RUBRIC))

    case = await db.get_agent_evaluation_case(report.cases[0].case_id)
    base = await db.get_source_unit_revision_projection(update.source_unit_id, update.base_unit_revision_id)
    assert case.manifest["derivation_id"] == update.id
    assert case.manifest["projection"] == source_projection_to_payload(update.projection)
    assert case.manifest["base_projection"] == source_projection_to_payload(base)
    first_case = await db.get_agent_evaluation_case(by_default.cases[0].case_id)
    assert first_case.manifest["derivation_id"] == first.id
    assert "base_projection" not in first_case.manifest


async def test_a_named_derivation_must_be_the_documents_own_with_a_readable_base(db) -> None:
    _first, update = await _updated_document(db)

    class BaseGone:
        get_document = db.get_document
        find_source_unit_by_document_id = db.find_source_unit_by_document_id
        list_source_derivation_attempts = db.list_source_derivation_attempts

        async def get_source_unit_revision_projection(self, _unit_id, _revision_id):
            return None

    unknown = await _pin(db, DerivationCaseRequest(doc_id=DOC_ID, derivation_id="sdrv-elsewhere", rubric=RUBRIC))
    gone = await _pin(db, DerivationCaseRequest(doc_id=DOC_ID, derivation_id=update.id, rubric=RUBRIC), BaseGone())

    assert (unknown.cases, unknown.skipped["derivation_not_found"]) == ((), (DOC_ID,))
    assert (gone.cases, gone.skipped["base_unavailable"]) == ((), (DOC_ID,))


async def test_replay_plans_an_update_against_its_pinned_base_as_production_did(db) -> None:
    first, update = await _updated_document(db)
    report = await _pin(db, DerivationCaseRequest(doc_id=DOC_ID, derivation_id=update.id, rubric=RUBRIC))
    case = await db.get_agent_evaluation_case(report.cases[0].case_id)
    planned = []

    async def plan(authority, work, _candidate):
        planned.append((authority, work.base))
        # The planner production staged the update with.
        return await fixture_request_planner(
            work.projection, access_context_hash=work.access_context_hash,
            extractor=ProjectionFragmentRecordingExtractor(),
        )(authority)

    requested = []

    async def extract(request, _work, _candidate):
        requested.append(request.id)
        return MemoryExtractionResult(memories=[])

    executor = SourceUnitDerivationReplayExecutor(plan, extract)
    output = await executor.execute(case, CANDIDATE)
    replayed_requests = list(requested)
    without_base = await executor.execute(
        replace(case, manifest={key: value for key, value in case.manifest.items() if key != "base_projection"}),
        CANDIDATE,
    )

    production_authority = plan_projection_evidence_work(
        update.projection,
        committed_base_snapshot=CommittedSourceUnitSnapshot(
            unit_revision=first.projection.source_unit_revisions[0],
            observation_revisions=first.projection.observation_revisions,
        ),
        reprocess_all_current_observations=False,
    )
    [(authority, base)] = planned
    assert authority == production_authority
    # Replay asks the same Claim Extraction requests production staged.
    assert replayed_requests and replayed_requests == [batch.batch_id for batch in update.batches]
    assert base.source_unit_revisions[0].id == update.base_unit_revision_id
    assert output["error_type"] is None
    # Without its base the update cannot be planned, and the case fails.
    assert without_base["error_type"] == "evidence_authority_planning_failed"
    ground_truth = await db.get_accepted_ground_truth_revision(report.cases[0].ground_truth_revision_id)
    checks = {
        check.criterion: (check.label.value, check.reason_code)
        for check in deterministic_checks(case, ground_truth, without_base)
    }
    assert checks["derivation_terminal_state"] == ("fail", "derivation_failed:INCREMENTAL_BASE_UNAVAILABLE")


async def test_an_extraction_error_is_raised_for_the_evaluator_to_record_as_an_error(db) -> None:
    await _synced_document(db)
    report = await _pin(db, DerivationCaseRequest(doc_id=DOC_ID, rubric=RUBRIC))
    case = await db.get_agent_evaluation_case(report.cases[0].case_id)

    async def plan(authority, work, _candidate):
        return await fixture_request_planner(work.projection, access_context_hash=work.access_context_hash)(authority)

    async def extract(_request, _work, _candidate):
        return MemoryExtractionResult(memories=[], error_type="unexpected_error")

    # Only a planning failure is a property of the pinned input; a provider fault is not cached as a result.
    with pytest.raises(ValueError):
        await SourceUnitDerivationReplayExecutor(plan, extract).execute(case, CANDIDATE)


class _AnswerExecutor:
    def __init__(self, memories: list[str], *, skipped_reading_groups: int = 0) -> None:
        self._memories = memories
        self._skipped = skipped_reading_groups

    async def execute(self, case: AgentEvaluationCase, _candidate: Mapping[str, object]) -> Mapping[str, object]:
        # Each Memory cites the case's first observation revision as a whole, as a valid answer does.
        revision = case.manifest["projection"]["observation_revisions"][0]
        content = str(revision["content"])
        selection = {
            "parts": [
                {
                    "role": "primary",
                    "kind": "text",
                    "raw_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "excerpt": content,
                    "presentation_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "anchor": {
                        "observation_id": revision["observation_id"],
                        "observation_revision_id": revision["id"],
                        "kind": "text_range",
                        "range_start": 0,
                        "range_end": len(content),
                    },
                }
            ]
        }
        return {
            "case_kind": case.case_kind.value,
            "extraction": {
                "memories": [
                    {"content": memory, "resolved_evidence_selection": selection} for memory in self._memories
                ],
                "evidence_telemetry": {"skipped_reading_group_count": self._skipped},
            },
            "error_type": None,
        }


async def _quick_run(db: Database, executor: _AnswerExecutor, rubric: Mapping[str, object]):
    await _synced_document(db)
    seeded = await seed_source_unit_derivation_cases(
        db, OfflineAgentEvaluation(db, executors={}),
        requests=[DerivationCaseRequest(doc_id=DOC_ID, rubric=rubric)],
        role=AgentEvaluationRole.SENTINEL, selection_policy_version="quick-v1", actor=ACTOR,
    )
    evaluation = OfflineAgentEvaluation(db, executors={AgentEvaluationCaseKind.SOURCE_UNIT_DERIVATION: executor})
    return await evaluation.execute_run(
        cohort_id=seeded.cohort_id, candidate_manifest=CANDIDATE,
        evaluator_suite="quick", evaluator_version="1", created_by=ACTOR,
    )


def _labels(report) -> dict[str, tuple[str, str]]:
    return {
        assessment.criterion: (str(assessment.label), assessment.reason_code)
        for assessment in report.assessments
        if assessment.annotator_kind == "code"
    }


async def test_a_derivation_that_reads_everything_passes_its_code_checks(db) -> None:
    report = await _quick_run(
        db, _AnswerExecutor(["Recovery uses the active extraction contract."]), RUBRIC,
    )

    assert _labels(report)["reading_group_coverage"] == ("pass", "reading_groups_read")
    assert report.verdict == "pass"


async def test_skipped_reading_groups_fail_the_run(db) -> None:
    report = await _quick_run(db, _AnswerExecutor([], skipped_reading_groups=2), RUBRIC)

    assert _labels(report)["reading_group_coverage"] == ("fail", "reading_groups_skipped:2")
    assert report.verdict == "fail"


async def test_what_the_memories_say_is_left_to_the_expectation_judge(db) -> None:
    report = await _quick_run(db, _AnswerExecutor([]), RUBRIC)

    assert set(_labels(report)) == {
        "typed_output", "fragment_selection_resolution", "derivation_terminal_state", "reading_group_coverage",
    }
    assert report.verdict == "pass"


@pytest.mark.parametrize(
    "rubric",
    [
        {"expected_memory_count": {"min": 1}, "required_terms": ["contract"]},
        {"expectations": []},
        {"expectations": ["", "A memory states the contract."]},
        {"expectations": "A memory states the contract."},
    ],
)
async def test_a_derivation_rubric_is_a_list_of_expectations(db, rubric) -> None:
    await _synced_document(db)

    with pytest.raises(ValueError, match="requires expectations"):
        await seed_source_unit_derivation_cases(
            db, OfflineAgentEvaluation(db, executors={}),
            requests=[DerivationCaseRequest(doc_id=DOC_ID, rubric=rubric)],
            role=AgentEvaluationRole.SENTINEL, selection_policy_version="quick-v1", actor=ACTOR,
        )


@pytest.mark.parametrize(
    ("results", "errors", "counts", "verdict"),
    [
        (1, 0, {"pass": 3}, "pending"),
        (2, 1, {"pass": 3}, "fail"),
        (2, 0, {"pass": 3, "fail": 1}, "fail"),
        (2, 0, {"pass": 3, "unknown": 1}, "unknown"),
        (2, 0, {"pass": 6}, "pass"),
    ],
)
def test_the_verdict_waits_for_every_result_and_lets_no_failure_or_doubt_pass(results, errors, counts, verdict) -> None:
    assert run_verdict(
        expected_result_count=2, result_count=results, error_result_count=errors, check_counts=counts,
    ) == verdict


async def test_the_seed_route_pins_cases_for_a_maintenance_operator_only(db, tmp_path) -> None:
    from memforge.server.admin_api import create_admin_app

    await _synced_document(db)
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    body = {
        "selection_policy_version": "quick-v1",
        "cases": [{"doc_id": DOC_ID, "rubric": RUBRIC, "population": "failure_regression"}],
    }
    responses = {}
    for operator in (False, True):
        app = create_admin_app(
            db=db, config=config, principal_resolver=lambda _request: ACTOR,
            workspace_role_resolver=lambda _request: "workspace_admin",
            maintenance_operator_resolver=lambda _request, operator=operator: operator,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            responses[operator] = await client.post("/api/v1/agent-evaluations/derivation-cases/seed", json=body)

    assert responses[False].status_code == 403
    seeded = responses[True].json()
    assert responses[True].status_code == 200, responses[True].text
    assert [(case["doc_id"], case["population"]) for case in seeded["cases"]] == [(DOC_ID, "failure_regression")]
    cohort = await db.get_agent_evaluation_cohort(seeded["cohort_id"])
    assert {item.role for item in cohort.items} == {AgentEvaluationRole.SENTINEL}


async def test_a_run_admitted_without_code_or_model_pins_the_serving_release_and_generation_model(
    db, tmp_path,
) -> None:
    from memforge.runtime import resolve_models
    from memforge.server.admin_api import create_admin_app

    await _synced_document(db)
    seeded = await seed_source_unit_derivation_cases(
        db, OfflineAgentEvaluation(db, executors={}),
        requests=[DerivationCaseRequest(doc_id=DOC_ID, rubric=RUBRIC)],
        role=AgentEvaluationRole.SENTINEL, selection_policy_version="quick-v1", actor=ACTOR,
    )
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    config.llm.enrichment_model = "openai/gpt-5-mini"
    config.llm.enrichment_api_key = "generation-key"
    app = create_admin_app(db=db, config=config, principal_resolver=lambda _request: ACTOR)
    candidate = {key: value for key, value in CANDIDATE.items() if key not in {"code_revision", "model"}}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        admitted = await client.post(
            "/api/v1/agent-evaluations/runs",
            json={"cohort_id": seeded.cohort_id, "candidate_manifest": candidate,
                  "evaluator_suite": "quick", "evaluator_version": "1"},
        )

    assert admitted.status_code == 202, admitted.text
    pinned = admitted.json()["run"]["candidate_manifest"]
    assert pinned["code_revision"] == current_release()
    assert pinned["model"] == (await resolve_models(db, config)).generation.route


def _report(verdict: str, *, state: str = "completed", assessments=(), results=()) -> dict:
    return {
        "run": {"run_id": "aer-1", "cohort_id": "aeo-1", "candidate_manifest": {"code_revision": "1.0", "model": "m"}},
        "execution": {"state": state},
        "summary": {"verdict": verdict, "completed_result_count": len(results), "error_result_count": 0},
        "results": list(results),
        "assessments": list(assessments),
    }


def test_waiting_reads_the_report_until_the_run_finishes_or_the_timeout_passes() -> None:
    # Every result is in before the last one is judged, so the verdict alone does not end the wait.
    reads = iter([_report("pending", state="queued"), _report("pass", state="leased"), _report("pass")])
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    finished = wait_for_report(lambda: next(reads), timeout_s=60, poll_s=10, sleep=sleep, clock=lambda: now[0])
    stuck = wait_for_report(
        lambda: _report("pending", state="queued"), timeout_s=25, poll_s=10, sleep=sleep, clock=lambda: now[0],
    )

    assert report_verdict(finished) == "pass"
    assert stuck is None
    assert report_verdict(_report("pending", state="failed")) == "fail"


def test_the_summary_counts_checks_and_lists_what_did_not_pass() -> None:
    report = _report(
        "fail",
        results=[{"result_id": "r1", "case_id": "aec-1", "status": "completed"}],
        assessments=[
            {"annotator_kind": "code", "criterion": "reading_group_coverage", "label": "fail",
             "reason_code": "reading_groups_skipped:2", "target_result_id": "r1"},
            {"annotator_kind": "code", "criterion": "typed_output", "label": "pass",
             "reason_code": "typed_output_valid", "target_result_id": "r1"},
            {"annotator_kind": "code", "criterion": "candidate_execution", "label": "needs_review",
             "reason_code": "candidate_artifact_unavailable", "target_result_id": "r1"},
        ],
    )

    summary = render_report_markdown(report)

    assert "- Verdict: **fail**" in summary
    assert "| reading_group_coverage | 0 | 1 | 0 |" in summary
    assert "| typed_output | 1 | 0 | 0 |" in summary
    # A check waiting for a person's review is counted as undecided.
    assert "| candidate_execution | 0 | 0 | 1 |" in summary
    assert "| `aec-1` | reading_group_coverage | fail: reading_groups_skipped:2 |" in summary


def _projection_reading_an_image(payload: bytes):
    from datetime import datetime, timezone

    from memforge.models import ContentItem, NormalizedContent, RawContent
    from memforge.pipeline.source_projection_adapters import project_source_item
    from memforge.source_artifacts import StoredSourceArtifact

    item = ContentItem(
        item_id="confluence-123", title="Page", source_url="https://example.test/123",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc), version="1",
        extra={"page_id": "123", "space_key": "ENG"},
    )
    return project_source_item(
        source_id="src-1", source_type="confluence", run_id="run-image", item=item,
        raw=RawContent(item=item, body=b"<h1>Page</h1>", content_type="text/html"),
        normalized=NormalizedContent(item=item, markdown_body="# Page"),
        artifacts=(
            StoredSourceArtifact(
                id="artifact-diagram", provider_key="attachment-1", parent_observation_type="page_body",
                parent_provider_key="123:body", provider_revision="1", filename="diagram.png",
                media_type="image/png", size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                uri="artifact://diagram-1.png", inference_eligible=True, inference_ineligible_reason=None,
            ),
        ),
    )


class _ArtifactStore:
    def __init__(self, stored: dict[str, bytes]) -> None:
        self.stored = stored

    def read_artifact(self, uri: str) -> bytes:
        return self.stored[uri]


@pytest.mark.parametrize(
    ("stored", "refusal"),
    [
        ({"artifact://diagram-1.png": b"PNG-bytes"}, None),
        ({}, "artifact_unavailable"),
        ({"artifact://diagram-1.png": b"changed"}, "artifact_integrity_failed"),
    ],
    ids=["stored_as_pinned", "deleted", "altered"],
)
def test_replay_reads_a_pinned_image_only_as_it_was_stored(stored, refusal) -> None:
    from memforge.evals.offline_evaluation import OfflineArtifactUnavailable, pinned_image_loader

    projection = _projection_reading_an_image(b"PNG-bytes")
    [image_observation] = [
        revision.observation_id
        for revision in projection.observation_revisions
        if "source_artifact" in revision.metadata
    ]
    load = pinned_image_loader(projection, _ArtifactStore(stored))

    if refusal is None:
        [image] = load({image_observation})
        assert image.body == b"PNG-bytes"
    else:
        with pytest.raises(OfflineArtifactUnavailable, match=refusal):
            load({image_observation})
