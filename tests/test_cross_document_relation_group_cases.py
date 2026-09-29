from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.evals.cross_document_relation_cases import (
    RelationCaseSkip,
    RelationGroupLabels,
    seed_cross_document_relation_group_cases,
)
from memforge.evals.offline_evaluation import (
    RELATION_LABEL_CRITERION,
    UNLABELLED_RELATION_CRITERION,
    AgentEvaluationCaseKind,
    CrossDocumentRelationGroupReplayExecutor,
    CrossDocumentRelationReplayExecutor,
    OfflineAgentEvaluation,
    agent_evaluation_report_public_payload,
)
from memforge.evals.offline_runtime import build_offline_evaluation_for_run
from memforge.evals.offline_worker import OfflineEvaluationWorker
from memforge.llm.structured import CrossDocumentRelationResponse
from memforge.memory.cross_document_relation import (
    CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
    CrossDocumentRelationLabel,
)
from memforge.models import Memory, content_hash
from memforge.storage.database import Database
from tests.llm_fixture import FIXTURE_MODEL, fixture_budget
from tests.relation_evidence_fixture import (
    primary_evidence_unit_fixture,
    primary_observation_revision_fixture,
)

ACTOR = "operator@example.test"
LABELLED_BY = "two reviewers, consensus only"
SOURCE_ID = "src-teams"
UPDATED_AT = datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc)
# mem-earlier's Evidence is recorded before its challenger's, so an ``updates``
# label on that pair is ordered by the Evidence times.
EARLIER_OBSERVED_AT = "2026-03-01T10:00:00.000+0000"
# Two candidates that repeat each other and say nothing about the subject.
REPEATED_STATEMENT = "The export job runs nightly."
CANDIDATE_MANIFEST = {
    "code_revision": "test",
    "prompt_hash": "test",
    "schema_version": "1",
    "contract_version": CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
    "model": FIXTURE_MODEL,
    "replay_harness_version": "1",
}


STATEMENTS = {
    "mem-subject": "Payroll runs weekly.",
    "mem-monthly": "Payroll runs monthly.",
    "mem-earlier": "Payroll ran fortnightly until March.",
    "mem-same-day": "Payroll runs every second week.",
    "mem-repeat-1": REPEATED_STATEMENT,
    "mem-repeat-2": REPEATED_STATEMENT,
    "mem-other": "The export job writes CSV files.",
    "mem-second": "Payslips are sent by email.",
    "mem-second-candidate": "Payslips are sent by post.",
    "mem-private": "Private note.",
    "mem-retired": "Retired statement.",
}


def _memory(memory_id: str, **overrides) -> Memory:
    content = STATEMENTS[memory_id]
    return replace(
        Memory(
            id=memory_id,
            memory_type="decision",
            content=content,
            content_hash=content_hash(content),
            updated_at=UPDATED_AT,
        ),
        **overrides,
    )


def _labelled_hashes(*memory_ids: str) -> dict[str, str]:
    """The content hash each Memory had when its group was labelled."""

    return {memory_id: content_hash(STATEMENTS.get(memory_id, memory_id)) for memory_id in memory_ids}


def _group(
    challenger_memory_id: str,
    candidate_memory_ids: tuple[str, ...],
    labels: dict[str, CrossDocumentRelationLabel],
) -> RelationGroupLabels:
    return RelationGroupLabels(
        challenger_memory_id=challenger_memory_id,
        candidate_memory_ids=candidate_memory_ids,
        labels=labels,
        content_hashes=_labelled_hashes(challenger_memory_id, *candidate_memory_ids),
    )


class _GroupStore:
    def __init__(self, db: Database, memories: list[Memory]) -> None:
        self.db = db
        self.memories = {memory.id: memory for memory in memories}
        self.units = {
            memory.id: (replace(primary_evidence_unit_fixture(memory.id), source_id=SOURCE_ID),)
            for memory in memories
        }
        self.observed_at: dict[str, str] = {}
        self.unit_reads: dict[str, int] = {}
        self.revision_reads: dict[str, int] = {}

    async def list_memories_by_ids(self, memory_ids):
        return [self.memories[memory_id] for memory_id in memory_ids if memory_id in self.memories]

    async def get_memory_evidence_units(self, memory_id):
        self.unit_reads[memory_id] = self.unit_reads.get(memory_id, 0) + 1
        return self.units.get(memory_id, ())

    async def get_document(self, doc_id):
        return SimpleNamespace(title=f"Title {doc_id}")

    async def get_current_source_observation_revisions(self, source_unit_id):
        self.revision_reads[source_unit_id] = self.revision_reads.get(source_unit_id, 0) + 1
        memory_id = source_unit_id.removeprefix("unit-")
        revision = primary_observation_revision_fixture(memory_id)
        if memory_id in self.observed_at:
            revision = replace(revision, observed_at=self.observed_at[memory_id])
        return {f"obs-{memory_id}": revision}

    async def get_source(self, source_id):
        return await self.db.get_source(source_id)


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "relation-group-cases.db"))
    await database.connect()
    await database.upsert_source(
        id=SOURCE_ID,
        type="teams",
        name="Teams",
        config_json="{}",
        owner_user_id="owner-1",
        access_policy="workspace",
    )
    yield database
    await database.close()


def _group_store(db: Database) -> _GroupStore:
    store = _GroupStore(
        db,
        [
            *(
                _memory(memory_id)
                for memory_id in STATEMENTS
                if memory_id not in {"mem-private", "mem-retired"}
            ),
            _memory("mem-private", visibility="private", owner_user_id="owner-1"),
            _memory("mem-retired", status="retired"),
        ],
    )
    store.observed_at["mem-earlier"] = EARLIER_OBSERVED_AT
    return store


def _payroll_group(**overrides) -> RelationGroupLabels:
    fields = {
        "challenger_memory_id": "mem-subject",
        "candidate_memory_ids": (
            "mem-monthly",
            "mem-repeat-1",
            "mem-earlier",
            "mem-repeat-2",
            "mem-same-day",
            "mem-other",
        ),
        "labels": {
            "mem-monthly": CrossDocumentRelationLabel.CONTRADICTS,
            "mem-earlier": CrossDocumentRelationLabel.UPDATES,
            "mem-same-day": CrossDocumentRelationLabel.UPDATES,
            "mem-repeat-1": CrossDocumentRelationLabel.NONE,
            "mem-repeat-2": CrossDocumentRelationLabel.NONE,
        },
        **overrides,
    }
    return _group(fields["challenger_memory_id"], fields["candidate_memory_ids"], fields["labels"])


def _second_group() -> RelationGroupLabels:
    return _group(
        "mem-second",
        ("mem-second-candidate",),
        {"mem-second-candidate": CrossDocumentRelationLabel.CONTRADICTS},
    )


async def _seed(db: Database, store: _GroupStore, groups: list[RelationGroupLabels]):
    return await seed_cross_document_relation_group_cases(
        store,
        OfflineAgentEvaluation(db, executors={}),
        groups=groups,
        labelled_by=LABELLED_BY,
        actor=ACTOR,
    )


def _questions(prompt: str) -> list[dict]:
    return json.loads(prompt.split("<questions>")[1].split("</questions>")[0])


class _RecordingClient:
    """Answers every question with ``label_for`` and records each request's prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def request_budget(self, model=None):
        return fixture_budget(input_tokens=100_000, output_tokens=64_000, correction_reserve=0)

    def request_fits(self, prompt, **_kwargs):
        return True

    def label_for(self, question: dict, questions: list[dict]) -> str:
        statement = question["candidate"]["statement"]
        return "contradicts" if "monthly" in statement else "none"

    async def classify_cross_document_relations(self, prompt, *, max_tokens, model=None):
        self.prompts.append(prompt)
        questions = _questions(prompt)
        return CrossDocumentRelationResponse(
            decisions=[
                {"pair_index": question["pair_index"], "label": self.label_for(question, questions)}
                for question in questions
            ]
        )


class _ContaminatedClient(_RecordingClient):
    """Judges a candidate equivalent to the subject when another candidate repeats it.

    This is the contamination grouped requests showed: candidates that repeat
    one another read as repeating the subject.
    """

    def label_for(self, question: dict, questions: list[dict]) -> str:
        statement = question["candidate"]["statement"]
        repeats = sum(other["candidate"]["statement"] == statement for other in questions)
        if repeats > 1:
            return "equivalent"
        return super().label_for(question, questions)


class _UnlabelledRelationClient(_RecordingClient):
    """Relates the unlabelled candidate mem-other, besides the base labels."""

    def label_for(self, question: dict, questions: list[dict]) -> str:
        if question["candidate"]["statement"] == "The export job writes CSV files.":
            return "equivalent"
        return super().label_for(question, questions)


@pytest.mark.asyncio
async def test_seed_pins_each_group_with_its_ordered_candidates_and_recorded_labels(db: Database) -> None:
    store = _group_store(db)
    groups = [_payroll_group(), _second_group()]

    report = await _seed(db, store, groups)
    again = await _seed(db, store, groups)

    assert report == again
    assert report.pinned_group_count == 2
    assert report.candidate_count == 7
    assert report.labelled_pair_count == 6
    # mem-same-day shares its Evidence date with the subject, so its updates
    # label is recorded as contradicts.
    assert report.label_counts == {"none": 2, "equivalent": 0, "updates": 1, "contradicts": 3}
    assert report.skipped == {reason.value: 0 for reason in RelationCaseSkip}
    assert report.skipped_groups == {}
    cohort = await db.get_agent_evaluation_cohort(report.cohort_id)
    assert cohort is not None and len(cohort.items) == 2
    cases = {}
    for item in cohort.items:
        case = await db.get_agent_evaluation_case(item.case_id)
        ground_truth = await db.get_accepted_ground_truth_revision(item.ground_truth_revision_id)
        cases[case.manifest["challenger"]["memory_id"]] = (case, ground_truth.rubric)
    case, rubric = cases["mem-subject"]
    assert case.case_kind is AgentEvaluationCaseKind.CROSS_DOCUMENT_RELATION_GROUP
    assert case.source_id == SOURCE_ID
    assert case.source_unit_id == "unit-mem-subject"
    assert case.manifest["classifier_version"] == CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION
    assert case.manifest["origin"] == {"labelled_by": LABELLED_BY}
    assert case.manifest["challenger"]["evidence"] == ["Evidence for mem-subject."]
    assert [candidate["memory_id"] for candidate in case.manifest["candidates"]] == list(
        _payroll_group().candidate_memory_ids
    )
    assert case.manifest["candidates"][2]["evidence_time"] == "2026-03-01"
    assert case.manifest["candidates"][0]["document_title"] == "Title doc-mem-monthly"
    assert rubric == {
        "expected_labels": {
            "mem-monthly": "contradicts",
            "mem-earlier": "updates",
            "mem-same-day": "contradicts",
            "mem-repeat-1": "none",
            "mem-repeat-2": "none",
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("missing-candidate", RelationCaseSkip.MEMORY_CHANGED),
        ("retired-candidate", RelationCaseSkip.MEMORY_CHANGED),
        ("candidate-edited-after-labelling", RelationCaseSkip.MEMORY_CHANGED),
        ("challenger-edited-after-labelling", RelationCaseSkip.MEMORY_CHANGED),
        ("private-candidate", RelationCaseSkip.PRIVATE_MEMORY),
        ("challenger-without-evidence", RelationCaseSkip.NO_SOURCE_EVIDENCE),
        ("candidate-from-changing-source", RelationCaseSkip.SOURCE_UNAVAILABLE),
        ("candidate-from-private-source", RelationCaseSkip.PRIVATE_MEMORY),
    ],
)
async def test_seed_skips_a_whole_group_it_cannot_show_and_pins_the_rest(
    db: Database, change: str, reason: RelationCaseSkip
) -> None:
    store = _group_store(db)
    candidates = _payroll_group().candidate_memory_ids
    if change == "missing-candidate":
        candidates = (*candidates, "mem-unknown")
    elif change == "retired-candidate":
        candidates = (*candidates, "mem-retired")
    elif change == "private-candidate":
        candidates = (*candidates, "mem-private")
    elif change.endswith("-edited-after-labelling"):
        edited = "mem-subject" if change.startswith("challenger") else "mem-earlier"
        content = "Payroll runs weekly on Fridays."
        store.memories[edited] = replace(store.memories[edited], content=content, content_hash=content_hash(content))
    elif change == "challenger-without-evidence":
        store.units["mem-subject"] = ()
    else:
        await db.upsert_source(
            id="src-unreadable",
            type="teams",
            name="Unreadable",
            config_json="{}",
            owner_user_id="someone-else",
            access_policy="private" if change == "candidate-from-private-source" else "workspace",
            access_state="changing" if change == "candidate-from-changing-source" else "active",
        )
        store.units["mem-other"] = (
            replace(primary_evidence_unit_fixture("mem-other"), source_id="src-unreadable"),
        )

    report = await _seed(db, store, [_payroll_group(candidate_memory_ids=candidates), _second_group()])

    assert report.skipped_groups == {"mem-subject": reason.value}
    assert report.skipped[reason.value] == 1
    assert report.pinned_group_count == 1
    assert report.label_counts == {"none": 0, "equivalent": 0, "updates": 0, "contradicts": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (
            lambda: [_payroll_group(labels={"mem-unlisted": CrossDocumentRelationLabel.NONE})],
            "does not list",
        ),
        (lambda: [_payroll_group(), _payroll_group()], "one group"),
        (
            lambda: [_payroll_group(candidate_memory_ids=("mem-monthly", "mem-monthly"))],
            "distinct candidates",
        ),
        (lambda: [_payroll_group(candidate_memory_ids=("mem-subject", "mem-monthly"))], "distinct candidates"),
        (
            lambda: [
                replace(
                    _second_group(),
                    content_hashes=_labelled_hashes("mem-second"),
                )
            ],
            "content hash",
        ),
        (
            lambda: [
                replace(
                    _second_group(),
                    content_hashes=_labelled_hashes("mem-second", "mem-second-candidate", "mem-monthly"),
                )
            ],
            "content hash",
        ),
    ],
    ids=[
        "unlisted-label",
        "repeated-challenger",
        "repeated-candidate",
        "challenger-as-candidate",
        "hash-missing",
        "hash-for-unlisted-memory",
    ],
)
async def test_seed_rejects_groups_whose_ids_do_not_fit(db: Database, groups, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        await _seed(db, _group_store(db), groups())


@pytest.mark.asyncio
async def test_seed_reads_each_memory_once_across_groups(db: Database) -> None:
    store = _group_store(db)
    shared_candidate = _group(
        "mem-second",
        ("mem-second-candidate", "mem-monthly"),
        {"mem-monthly": CrossDocumentRelationLabel.NONE},
    )

    report = await _seed(db, store, [_payroll_group(), shared_candidate])

    assert report.pinned_group_count == 2
    assert store.unit_reads["mem-monthly"] == 1
    assert set(store.unit_reads.values()) == {1}
    assert set(store.revision_reads.values()) == {1}


@pytest.mark.asyncio
async def test_replay_asks_about_every_candidate_of_a_group_in_one_request(db: Database) -> None:
    report = await _seed(db, _group_store(db), [_payroll_group()])
    cohort = await db.get_agent_evaluation_cohort(report.cohort_id)
    case = await db.get_agent_evaluation_case(cohort.items[0].case_id)
    client = _RecordingClient()

    output = await CrossDocumentRelationGroupReplayExecutor(client).execute(case, {"model": FIXTURE_MODEL})

    [prompt] = client.prompts
    assert prompt.count("Payroll runs weekly.") == 1
    assert [question["candidate"]["statement"] for question in _questions(prompt)] == [
        candidate["statement"] for candidate in case.manifest["candidates"]
    ]
    assert output["llm_calls"] == 1
    assert [(judgment["memory_id"], judgment["label"]) for judgment in output["judgments"]] == [
        ("mem-monthly", "contradicts"),
        ("mem-repeat-1", "none"),
        ("mem-earlier", "none"),
        ("mem-repeat-2", "none"),
        ("mem-same-day", "none"),
        ("mem-other", "none"),
    ]


async def _run(db: Database, cohort_id: str, client: _RecordingClient):
    evaluation = OfflineAgentEvaluation(
        db,
        executors={
            AgentEvaluationCaseKind.CROSS_DOCUMENT_RELATION_GROUP: CrossDocumentRelationGroupReplayExecutor(client)
        },
    )
    return await evaluation.execute_run(
        cohort_id=cohort_id,
        candidate_manifest=CANDIDATE_MANIFEST,
        evaluator_suite="cross_document_relation",
        evaluator_version="1",
        created_by=ACTOR,
    )


@pytest.mark.asyncio
async def test_a_group_run_scores_labelled_candidates_and_counts_unlabelled_relations(db: Database) -> None:
    seed = await _seed(db, _group_store(db), [_payroll_group(), _second_group()])

    report = await _run(db, seed.cohort_id, _UnlabelledRelationClient())

    assert report.completed_result_count == 2
    assert report.label_confusion == {
        "contradicts:contradicts": 1,
        "contradicts:none": 2,
        "updates:none": 1,
        "none:none": 2,
    }
    assert report.label_metrics["contradicts"]["recall"] == 1 / 3
    assert report.relation_summary == {
        "labelled_pairs": 6,
        "false_relations": 0,
        "none_recall": 1.0,
        "unlabelled_relations": 1,
    }
    code_checks = [assessment for assessment in report.assessments if assessment.annotator_kind == "code"]
    assert sum(check.criterion == RELATION_LABEL_CRITERION for check in code_checks) == 6
    assert {check.reason_code for check in code_checks if check.criterion == RELATION_LABEL_CRITERION} == {
        "contradicts:contradicts/mem-monthly",
        "none:none/mem-repeat-1",
        "updates:none/mem-earlier",
        "none:none/mem-repeat-2",
        "contradicts:none/mem-same-day",
        "contradicts:none/mem-second-candidate",
    }
    [unlabelled] = [check for check in code_checks if check.criterion == UNLABELLED_RELATION_CRITERION]
    assert (unlabelled.label, unlabelled.reason_code) == ("needs_review", "equivalent/mem-other")
    summary = agent_evaluation_report_public_payload(report, None)["summary"]
    assert summary["relation_summary"] == report.relation_summary


@pytest.mark.asyncio
async def test_a_group_without_labels_is_replayed_for_its_unlabelled_relations(db: Database) -> None:
    unlabelled = _group("mem-second", ("mem-second-candidate", "mem-monthly"), {})
    seed = await _seed(db, _group_store(db), [unlabelled])

    report = await _run(db, seed.cohort_id, _RecordingClient())

    assert (seed.pinned_group_count, seed.labelled_pair_count) == (1, 0)
    assert report.relation_summary == {
        "labelled_pairs": 0,
        "false_relations": 0,
        "none_recall": None,
        "unlabelled_relations": 1,
    }


@pytest.mark.asyncio
async def test_a_group_run_reports_candidates_judged_by_each_other_as_false_relations(db: Database) -> None:
    seed = await _seed(db, _group_store(db), [_payroll_group()])

    report = await _run(db, seed.cohort_id, _ContaminatedClient())

    assert report.label_confusion["none:equivalent"] == 2
    assert report.relation_summary["false_relations"] == 2
    assert report.relation_summary["none_recall"] == 0.0
    assert report.label_metrics["equivalent"]["precision"] == 0.0
    # Asked about on its own, a repeated candidate has nothing to repeat, so a
    # single-pair case cannot show this error.
    cohort = await db.get_agent_evaluation_cohort(seed.cohort_id)
    case = await db.get_agent_evaluation_case(cohort.items[0].case_id)
    alone = replace(
        case,
        manifest={
            "classifier_version": case.manifest["classifier_version"],
            "challenger": case.manifest["challenger"],
            "candidate": case.manifest["candidates"][1],
        },
    )
    single = await CrossDocumentRelationReplayExecutor(_ContaminatedClient()).execute(alone, {"model": FIXTURE_MODEL})
    assert single["label"] == "none"


@pytest.mark.asyncio
async def test_the_run_worker_replays_group_cases(db: Database, tmp_path) -> None:
    seed = await _seed(db, _group_store(db), [_payroll_group()])
    client = _RecordingClient()
    provider = SimpleNamespace(build_structured_llm_client=lambda _llm, *, max_concurrent: client)
    config = AppConfig(base_dir=tmp_path / "memforge")
    run, _execution = await OfflineAgentEvaluation(db, executors={}).admit_run(
        cohort_id=seed.cohort_id,
        candidate_manifest=CANDIDATE_MANIFEST,
        evaluator_suite="cross_document_relation",
        evaluator_version="1",
        created_by=ACTOR,
    )

    async def evaluation_factory(evaluation_run):
        return await build_offline_evaluation_for_run(db, config, provider, evaluation_run)

    execution = await OfflineEvaluationWorker(db, evaluation_factory=evaluation_factory, worker_id="worker").run_once()

    assert execution is not None
    report = await OfflineAgentEvaluation(db, executors={}).read_report(run.run_id, requesting_user_id=ACTOR)
    assert report.completed_result_count == 1
    assert report.relation_summary["labelled_pairs"] == 5
    assert len(client.prompts) == 1


def _app(db: Database, tmp_path, *, operator: bool):
    from memforge.server.admin_api import create_admin_app

    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    return create_admin_app(
        db=db,
        config=config,
        principal_resolver=lambda _request: ACTOR,
        workspace_role_resolver=lambda _request: "workspace_admin",
        maintenance_operator_resolver=lambda _request: operator,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("operator", "status"), [(False, 403), (True, 200)])
async def test_group_seed_route_requires_a_maintenance_operator(
    db: Database, tmp_path, operator: bool, status: int
) -> None:
    body = {
        "labelled_by": LABELLED_BY,
        "groups": [
            {
                "challenger_memory_id": "mem-unknown",
                "candidate_memory_ids": ["mem-a", "mem-b"],
                "labels": {"mem-a": "equivalent"},
                "content_hashes": _labelled_hashes("mem-unknown", "mem-a", "mem-b"),
            }
        ],
    }

    with TestClient(_app(db, tmp_path, operator=operator)) as client:
        response = client.post("/api/v1/agent-evaluations/relation-group-cases/seed", json=body)
        unlisted = client.post(
            "/api/v1/agent-evaluations/relation-group-cases/seed",
            json={**body, "groups": [{**body["groups"][0], "labels": {"mem-c": "none"}}]},
        )
        unknown_label = client.post(
            "/api/v1/agent-evaluations/relation-group-cases/seed",
            json={**body, "groups": [{**body["groups"][0], "labels": {"mem-a": "similar"}}]},
        )
        without_hashes = client.post(
            "/api/v1/agent-evaluations/relation-group-cases/seed",
            json={
                **body,
                "groups": [{key: value for key, value in body["groups"][0].items() if key != "content_hashes"}],
            },
        )

    assert response.status_code == status
    if not operator:
        return
    assert response.json() == {
        "cohort_id": None,
        "pinned_group_count": 0,
        "candidate_count": 0,
        "labelled_pair_count": 0,
        "label_counts": {"none": 0, "equivalent": 0, "updates": 0, "contradicts": 0},
        "skipped": {"memory_changed": 1, "private_memory": 0, "source_unavailable": 0, "no_source_evidence": 0},
        "skipped_groups": {"mem-unknown": "memory_changed"},
    }
    assert unlisted.status_code == 400
    assert unknown_label.status_code == 422
    assert without_hashes.status_code == 422
