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
    run_verdict,
)
from memforge.evals.offline_run import render_report_markdown, report_verdict, wait_for_report
from memforge.evals.source_unit_derivation_cases import (
    DerivationCaseRequest,
    seed_source_unit_derivation_cases,
)
from memforge.release import current_release
from memforge.source_derivation import source_unit_derivation_context_to_payload
from memforge.source_projection import source_projection_to_payload
from memforge.storage.database import Database
from tests.test_sync_bookkeeping import _stage_completed_v9_recovery_attempt

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
RUBRIC = {"expected_memory_count": {"min": 1}, "required_terms": ["extraction contract"]}


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


async def test_a_derivation_that_reads_everything_and_holds_the_rubric_passes(db) -> None:
    report = await _quick_run(
        db, _AnswerExecutor(["Recovery uses the active extraction contract."]), RUBRIC,
    )

    labels = _labels(report)
    assert labels["reading_group_coverage"] == ("pass", "reading_groups_read")
    assert labels["memory_count"] == ("pass", "memory_count:1")
    assert labels["required_terms"] == ("pass", "required_terms_missing:0/1")
    assert report.verdict == "pass"


async def test_skipped_reading_groups_and_missing_memories_fail_the_run(db) -> None:
    report = await _quick_run(db, _AnswerExecutor([], skipped_reading_groups=2), RUBRIC)

    labels = _labels(report)
    assert labels["reading_group_coverage"] == ("fail", "reading_groups_skipped:2")
    assert labels["memory_count"] == ("fail", "memory_count:0")
    assert labels["required_terms"] == ("fail", "required_terms_missing:1/1")
    assert report.verdict == "fail"


async def test_a_document_with_nothing_to_remember_passes_with_no_memories(db) -> None:
    report = await _quick_run(db, _AnswerExecutor([]), {"expected_memory_count": {"min": 0, "max": 0}})

    assert _labels(report)["memory_count"] == ("pass", "memory_count:0")
    assert "required_terms" not in _labels(report)
    assert report.verdict == "pass"


@pytest.mark.parametrize(
    ("rubric", "message"),
    [
        ({"expected_memory_count": {"min": 2, "max": 1}}, "min cannot exceed max"),
        ({"expected_memory_count": {"min": -1}}, "non-negative integers"),
        ({"expected_memory_count": {"least": 1}}, "takes min and max"),
        ({"required_terms": ["", "contract"]}, "non-empty strings"),
    ],
)
async def test_a_derivation_rubric_must_be_checkable(db, rubric, message) -> None:
    await _synced_document(db)

    with pytest.raises(ValueError, match=message):
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
    reads = iter([_report("pending", state="queued"), _report("pending", state="leased"), _report("pass")])
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
            {"annotator_kind": "code", "criterion": "memory_count", "label": "fail",
             "reason_code": "memory_count:0", "target_result_id": "r1"},
            {"annotator_kind": "code", "criterion": "typed_output", "label": "pass",
             "reason_code": "typed_output_valid", "target_result_id": "r1"},
            {"annotator_kind": "code", "criterion": "candidate_execution", "label": "needs_review",
             "reason_code": "candidate_artifact_unavailable", "target_result_id": "r1"},
        ],
    )

    summary = render_report_markdown(report)

    assert "- Verdict: **fail**" in summary
    assert "| memory_count | 0 | 1 | 0 |" in summary
    assert "| typed_output | 1 | 0 | 0 |" in summary
    # A check waiting for a person's review is counted as undecided.
    assert "| candidate_execution | 0 | 0 | 1 |" in summary
    assert "| `aec-1` | memory_count | fail: memory_count:0 |" in summary


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
