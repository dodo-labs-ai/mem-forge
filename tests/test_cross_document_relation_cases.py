from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.evals.cross_document_relation_cases import (
    RelationCaseSkip,
    seed_cross_document_relation_cases,
)
from memforge.evals.offline_evaluation import (
    AgentEvaluationCaseKind,
    AgentEvaluationPopulation,
    CrossDocumentRelationReplayExecutor,
    OfflineAgentEvaluation,
    relation_label_metrics,
)
from memforge.llm.structured import CrossDocumentRelationResponse
from memforge.memory.cross_source_conflict_reviews import CROSS_SOURCE_CONFLICT_REVIEW_KIND
from memforge.memory.cross_document_relation import (
    CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
    CrossDocumentRelationLabel,
)
from memforge.models import Memory, MemoryReview, content_hash
from memforge.storage.database import Database
from tests.llm_fixture import FIXTURE_MODEL, fixture_budget
from tests.relation_evidence_fixture import (
    primary_evidence_unit_fixture,
    primary_observation_revision_fixture,
)

UPDATED_AT = datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc)
ACTOR = "operator@example.test"


def _memory(memory_id: str, content: str, **overrides) -> Memory:
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


def _review(review_id: str, status: str, challenger: str, incumbent: str) -> MemoryReview:
    return MemoryReview(
        id=review_id,
        kind=CROSS_SOURCE_CONFLICT_REVIEW_KIND,
        status=status,
        incumbent_memory_id=incumbent,
        challenger_memory_id=challenger,
        reviewer="reviewer@example.test",
        expected_incumbent_updated_at=UPDATED_AT.isoformat(),
        expected_challenger_updated_at=UPDATED_AT.isoformat(),
        resolved_at=UPDATED_AT,
    )


class _ReviewStore:
    def __init__(self, db: Database, memories: list[Memory], reviews: list[MemoryReview]) -> None:
        self.db = db
        self.memories = {memory.id: memory for memory in memories}
        self.reviews = reviews
        self.units = {memory.id: (replace(primary_evidence_unit_fixture(memory.id), source_id="src-teams"),) for memory in memories}

    async def list_memory_reviews(self, status=None, kind=None, limit=100, offset=0):
        matching = [review for review in self.reviews if review.status == status and review.kind == kind]
        return matching[offset : offset + limit]

    async def list_memories_by_ids(self, memory_ids):
        return [self.memories[memory_id] for memory_id in memory_ids if memory_id in self.memories]

    async def get_memory_evidence_units(self, memory_id):
        return self.units.get(memory_id, ())

    async def get_document(self, doc_id):
        return type("Doc", (), {"title": f"Title {doc_id}"})()

    async def get_current_source_observation_revisions(self, source_unit_id):
        memory_id = source_unit_id.removeprefix("unit-")
        return {f"obs-{memory_id}": primary_observation_revision_fixture(memory_id)}

    async def get_source(self, source_id):
        return await self.db.get_source(source_id)


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "relation-cases.db"))
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


def _review_store(db: Database) -> _ReviewStore:
    memories = [
        _memory("mem-a", "Payroll runs weekly."),
        _memory("mem-b", "Payroll runs monthly."),
        _memory("mem-c", "The enum is called RETRO."),
        _memory("mem-d", "The enum is called RETRO_CHAIN."),
        _memory("mem-e", "Ticket 12 fixes the rounding bug."),
        _memory("mem-f", "Ticket 12 fixes the export bug."),
        _memory("mem-g", "Private note.", visibility="private", owner_user_id="owner-1"),
        _memory("mem-h", "Changed statement.", updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc)),
    ]
    store = _ReviewStore(
        db,
        memories,
        [
            _review("rev-confirmed", "approved", "mem-a", "mem-b"),
            _review("rev-updated", "approved", "mem-c", "mem-d"),
            _review("rev-dismissed", "rejected", "mem-e", "mem-f"),
            _review("rev-private", "rejected", "mem-g", "mem-a"),
            _review("rev-changed", "approved", "mem-h", "mem-a"),
            _review("rev-pending", "pending", "mem-a", "mem-f"),
        ],
    )
    return store


@pytest.mark.asyncio
async def test_seed_pins_decided_reviews_with_labels_and_is_repeatable(db: Database) -> None:
    store = _review_store(db)
    evaluation = OfflineAgentEvaluation(db, executors={})
    overrides = {"rev-updated": CrossDocumentRelationLabel.UPDATES}

    report = await seed_cross_document_relation_cases(store, evaluation, actor=ACTOR, label_overrides=overrides)
    again = await seed_cross_document_relation_cases(store, evaluation, actor=ACTOR, label_overrides=overrides)

    assert report == again
    assert report.pinned_case_count == 3
    assert report.label_counts == {"none": 1, "equivalent": 0, "updates": 1, "contradicts": 1}
    assert report.skipped == {
        RelationCaseSkip.MEMORY_CHANGED.value: 1,
        RelationCaseSkip.PRIVATE_MEMORY.value: 1,
        RelationCaseSkip.SOURCE_UNAVAILABLE.value: 0,
        RelationCaseSkip.NO_SOURCE_EVIDENCE.value: 0,
    }
    cohort = await db.get_agent_evaluation_cohort(report.cohort_id)
    assert cohort is not None and len(cohort.items) == 3
    labels_by_review = {}
    for item in cohort.items:
        case = await db.get_agent_evaluation_case(item.case_id)
        ground_truth = await db.get_accepted_ground_truth_revision(item.ground_truth_revision_id)
        assert case.case_kind is AgentEvaluationCaseKind.CROSS_DOCUMENT_RELATION
        assert case.source_id == "src-teams"
        assert case.manifest["classifier_version"] == CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION
        challenger_id = case.manifest["challenger"]["memory_id"]
        assert case.manifest["challenger"]["statement"] == store.memories[challenger_id].content
        assert case.manifest["challenger"]["evidence"] == [f"Evidence for {challenger_id}."]
        assert case.manifest["challenger"]["evidence_time"] == "2026-03-25"
        assert case.manifest["candidate"]["document_title"].startswith("Title ")
        labels_by_review[case.manifest["origin"]["review_id"]] = (
            ground_truth.rubric["expected_label"],
            item.population,
        )
    assert labels_by_review == {
        "rev-confirmed": ("contradicts", AgentEvaluationPopulation.REPRESENTATIVE_CONTROL),
        "rev-updated": ("updates", AgentEvaluationPopulation.REPRESENTATIVE_CONTROL),
        "rev-dismissed": ("none", AgentEvaluationPopulation.FAILURE_REGRESSION),
    }


@pytest.mark.asyncio
async def test_seed_skips_a_pair_without_source_evidence_and_rejects_unknown_relabels(db: Database) -> None:
    store = _review_store(db)
    store.units["mem-a"] = ()
    evaluation = OfflineAgentEvaluation(db, executors={})

    report = await seed_cross_document_relation_cases(store, evaluation, actor=ACTOR, label_overrides={})

    assert report.skipped[RelationCaseSkip.NO_SOURCE_EVIDENCE.value] == 1
    with pytest.raises(ValueError, match="rev-pending"):
        await seed_cross_document_relation_cases(
            store,
            evaluation,
            actor=ACTOR,
            label_overrides={"rev-pending": CrossDocumentRelationLabel.UPDATES},
        )


UNREADABLE_SOURCE_ID = "src-unreadable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("shown_memory_id", "source", "reason"),
    [
        # rev-confirmed pairs challenger mem-a with candidate mem-b.
        ("mem-a", {"access_policy": "private", "owner_user_id": "someone-else"}, RelationCaseSkip.PRIVATE_MEMORY),
        ("mem-b", {"access_policy": "private", "owner_user_id": "someone-else"}, RelationCaseSkip.PRIVATE_MEMORY),
        ("mem-a", {"access_policy": "workspace", "access_state": "changing"}, RelationCaseSkip.SOURCE_UNAVAILABLE),
        ("mem-b", {"access_policy": "workspace", "access_state": "changing"}, RelationCaseSkip.SOURCE_UNAVAILABLE),
        ("mem-a", None, RelationCaseSkip.SOURCE_UNAVAILABLE),
        ("mem-b", None, RelationCaseSkip.SOURCE_UNAVAILABLE),
    ],
    ids=[
        "private-challenger",
        "private-candidate",
        "changing-challenger",
        "changing-candidate",
        "missing-challenger",
        "missing-candidate",
    ],
)
async def test_seed_skips_a_pair_shown_from_an_unreadable_or_missing_source(
    db: Database,
    shown_memory_id: str,
    source: dict[str, str] | None,
    reason: RelationCaseSkip,
) -> None:
    if source is not None:
        await db.upsert_source(
            id=UNREADABLE_SOURCE_ID,
            type="teams",
            name="Unreadable",
            config_json="{}",
            owner_user_id=source.get("owner_user_id", "owner-1"),
            access_policy=source["access_policy"],
            access_state=source.get("access_state", "active"),
        )
    store = _review_store(db)
    store.units[shown_memory_id] = (
        replace(primary_evidence_unit_fixture(shown_memory_id), source_id=UNREADABLE_SOURCE_ID),
    )

    report = await seed_cross_document_relation_cases(
        store,
        OfflineAgentEvaluation(db, executors={}),
        actor=ACTOR,
        label_overrides={},
    )

    expected_skips = {
        RelationCaseSkip.MEMORY_CHANGED.value: 1,
        RelationCaseSkip.PRIVATE_MEMORY.value: 1,
        RelationCaseSkip.SOURCE_UNAVAILABLE.value: 0,
        RelationCaseSkip.NO_SOURCE_EVIDENCE.value: 0,
    }
    expected_skips[reason.value] += 1
    assert report.skipped == expected_skips
    assert report.pinned_case_count == 2
    cohort = await db.get_agent_evaluation_cohort(report.cohort_id)
    pinned_reviews = set()
    for item in cohort.items:
        case = await db.get_agent_evaluation_case(item.case_id)
        pinned_reviews.add(case.manifest["origin"]["review_id"])
        assert shown_memory_id not in {case.manifest["challenger"]["memory_id"], case.manifest["candidate"]["memory_id"]}
    assert pinned_reviews == {"rev-updated", "rev-dismissed"}


class _LabelClient:
    """Labels a pair contradicts when its challenger mentions weekly, otherwise none."""

    def request_budget(self, model=None):
        return fixture_budget(input_tokens=100_000, output_tokens=64_000, correction_reserve=0)

    def request_fits(self, prompt, **_kwargs):
        return True

    async def classify_cross_document_relations(self, prompt, *, max_tokens, model=None):
        label = "contradicts" if "weekly" in prompt else "none"
        return CrossDocumentRelationResponse(decisions=[{"pair_index": 0, "label": label, "reason": "fixture"}])


@pytest.mark.asyncio
async def test_relation_run_reports_label_precision_and_recall(db: Database) -> None:
    store = _review_store(db)
    seed = await seed_cross_document_relation_cases(
        store,
        OfflineAgentEvaluation(db, executors={}),
        actor=ACTOR,
        label_overrides={"rev-updated": CrossDocumentRelationLabel.UPDATES},
    )
    evaluation = OfflineAgentEvaluation(
        db,
        executors={AgentEvaluationCaseKind.CROSS_DOCUMENT_RELATION: CrossDocumentRelationReplayExecutor(_LabelClient())},
    )

    report = await evaluation.execute_run(
        cohort_id=seed.cohort_id,
        candidate_manifest={
            "code_revision": "test",
            "prompt_hash": "test",
            "schema_version": "1",
            "contract_version": CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
            "model": FIXTURE_MODEL,
            "replay_harness_version": "1",
        },
        evaluator_suite="cross_document_relation",
        evaluator_version="1",
        created_by=ACTOR,
    )

    assert report.completed_result_count == 3
    assert report.label_confusion == {"contradicts:contradicts": 1, "updates:none": 1, "none:none": 1}
    assert report.label_metrics["contradicts"]["precision"] == 1.0
    assert report.label_metrics["none"]["precision"] == 0.5
    assert report.label_metrics["updates"]["recall"] == 0.0
    assert report.label_metrics["equivalent"]["recall"] is None


def test_relation_label_metrics_ignores_codes_outside_the_label_set() -> None:
    metrics, confusion = relation_label_metrics(["none:none", "updates:contradicts", "candidate_execution_error"])

    assert confusion == {"none:none": 1, "updates:contradicts": 1}
    assert metrics["contradicts"] == {"expected": 0, "predicted": 1, "correct": 0, "precision": 0.0, "recall": None}


@pytest.mark.asyncio
@pytest.mark.parametrize(("operator", "status"), [(False, 403), (True, 200)])
async def test_seed_route_requires_a_maintenance_operator(db: Database, tmp_path, operator: bool, status: int) -> None:
    from memforge.server.admin_api import create_admin_app

    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    app = create_admin_app(
        db=db,
        config=config,
        principal_resolver=lambda _request: ACTOR,
        workspace_role_resolver=lambda _request: "workspace_admin",
        maintenance_operator_resolver=lambda _request: operator,
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/agent-evaluations/relation-cases/seed", json={"label_overrides": {}})
        unknown = client.post(
            "/api/v1/agent-evaluations/relation-cases/seed",
            json={"label_overrides": {"rev-missing": "updates"}},
        )

    assert response.status_code == status
    if operator:
        assert response.json() == {
            "cohort_id": None,
            "pinned_case_count": 0,
            "label_counts": {"none": 0, "equivalent": 0, "updates": 0, "contradicts": 0},
            "skipped": {"memory_changed": 0, "private_memory": 0, "source_unavailable": 0, "no_source_evidence": 0},
        }
        assert unknown.status_code == 400


async def _executed_relation_run(db: Database) -> str:
    seed = await seed_cross_document_relation_cases(
        _review_store(db),
        OfflineAgentEvaluation(db, executors={}),
        actor=ACTOR,
        label_overrides={},
    )
    evaluation = OfflineAgentEvaluation(
        db,
        executors={AgentEvaluationCaseKind.CROSS_DOCUMENT_RELATION: CrossDocumentRelationReplayExecutor(_LabelClient())},
    )
    report = await evaluation.execute_run(
        cohort_id=seed.cohort_id,
        candidate_manifest={
            "code_revision": "test",
            "prompt_hash": "test",
            "schema_version": "1",
            "contract_version": CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
            "model": FIXTURE_MODEL,
            "replay_harness_version": "1",
        },
        evaluator_suite="cross_document_relation",
        evaluator_version="1",
        created_by=ACTOR,
    )
    return report.run.run_id


@pytest.mark.asyncio
async def test_a_case_pinned_for_another_classifier_is_neither_curated_nor_replayed(db: Database) -> None:
    run_id = await _executed_relation_run(db)
    [output, *_rest] = await OfflineAgentEvaluation(db, executors={}).read_case_outputs(
        run_id, requesting_user_id=ACTOR
    )
    earlier = {**output.case.manifest, "classifier_version": "cross-document-relation-v1"}

    with pytest.raises(ValueError, match="cross-document-relation-v1"):
        await OfflineAgentEvaluation(db, executors={}).curate_case(
            case_kind=AgentEvaluationCaseKind.CROSS_DOCUMENT_RELATION,
            source_id=output.case.source_id,
            doc_id=output.case.doc_id,
            source_unit_id=output.case.source_unit_id,
            manifest=earlier,
            promotion_policy_version="test",
            created_by=ACTOR,
        )
    with pytest.raises(ValueError, match="cross-document-relation-v1"):
        await CrossDocumentRelationReplayExecutor(_LabelClient()).execute(
            replace(output.case, manifest=earlier),
            {"model": FIXTURE_MODEL},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("operator", "status"), [(False, 403), (True, 200)])
async def test_case_outputs_route_returns_pinned_input_label_and_reason(
    db: Database, tmp_path, operator: bool, status: int
) -> None:
    from memforge.server.admin_api import create_admin_app

    run_id = await _executed_relation_run(db)
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    app = create_admin_app(
        db=db,
        config=config,
        principal_resolver=lambda _request: ACTOR,
        workspace_role_resolver=lambda _request: "workspace_admin",
        maintenance_operator_resolver=lambda _request: operator,
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v1/agent-evaluations/runs/{run_id}/case-outputs")
        missing = client.get("/api/v1/agent-evaluations/runs/aer-missing/case-outputs")

    assert response.status_code == status
    if not operator:
        return
    assert missing.status_code == 404
    payload = response.json()
    assert payload["run_id"] == run_id
    by_review = {case["manifest"]["origin"]["review_id"]: case for case in payload["cases"]}
    assert set(by_review) == {"rev-confirmed", "rev-updated", "rev-dismissed"}
    confirmed = by_review["rev-confirmed"]
    assert confirmed["rubric"] == {"expected_label": "contradicts"}
    assert confirmed["manifest"]["challenger"]["evidence"] == ["Evidence for mem-a."]
    assert confirmed["result"]["output"] == {
        "case_kind": AgentEvaluationCaseKind.CROSS_DOCUMENT_RELATION.value,
        "classifier_version": CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
        "label": "contradicts",
        "reason": "fixture",
    }
    assert {check["reason_code"] for check in confirmed["checks"]} >= {"contradicts:contradicts"}
