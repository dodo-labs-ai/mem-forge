"""Hand-written reconciliation cases: imported against a synced Document, then replayed.

The fixture client stands in for the model; this is contract coverage, not
model-accuracy evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memforge.evals.offline_evaluation import (
    AcceptedGroundTruthRevision,
    AgentEvaluationRole,
    DeterministicCheckLabel,
    OfflineAgentEvaluation,
    SourceUnitReconciliationReplayExecutor,
    deterministic_checks,
)
from memforge.evals.reconciliation_cases import ReconciliationCaseRequest, seed_reconciliation_cases
from memforge.models import DocumentRecord
from memforge.storage.database import Database
from tests.test_claim_revision import Client

ACTOR = "dev"
DOCUMENT = DocumentRecord(
    doc_id="doc-release", source="source-1", source_url="https://example.test/release", title="Release process",
    space_or_project="TEST", author=None, last_modified=datetime(2026, 10, 1, tzinfo=timezone.utc),
    labels=[], version="2", content_hash="hash", token_count=20, last_synced=datetime(2026, 10, 1, tzinfo=timezone.utc),
)
FRIDAY = {"id": "friday", "claim": "Production releases ship every Friday.", "memory_type": "convention"}


def situation(candidate_claim: str, *, support="supported") -> dict[str, object]:
    return {
        "candidates": [{
            "claim": candidate_claim,
            "memory_type": "convention",
            "evidence": [{"role": "primary", "excerpt": candidate_claim, "observation": "body", "revision": "v2"}],
        }],
        "incumbents": [FRIDAY],
        "support": {"friday": support},
    }


class Documents:
    """The synced Documents a case may be imported against."""

    async def get_document(self, doc_id):
        return DOCUMENT if doc_id == DOCUMENT.doc_id else None

    async def list_documents(self, source=None, search=None, limit=100):
        return [DOCUMENT]

    async def find_source_unit_by_document_id(self, source_id, document_id, *, current_only=False):
        return SimpleNamespace(id="unit-release") if document_id == DOCUMENT.doc_id else None


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "reconciliation-cases.db"))
    await database.connect()
    await database.upsert_source(
        id="source-1", type="confluence", name="Reconciliation cases", config_json="{}",
        access_policy="workspace", owner_user_id="fixture-user",
    )
    yield database
    await database.close()


async def _import(db, request: ReconciliationCaseRequest):
    return await seed_reconciliation_cases(
        Documents(), OfflineAgentEvaluation(db, executors={}), requests=[request],
        role=AgentEvaluationRole.SENTINEL, selection_policy_version="reconciliation-v1", actor=ACTOR,
    )


async def _replayed(db, case_input, client, rubric):
    report = await _import(db, ReconciliationCaseRequest(input=case_input, rubric=rubric, doc_id=DOCUMENT.doc_id))
    case = await db.get_agent_evaluation_case(report.cases[0].case_id)
    output = await SourceUnitReconciliationReplayExecutor(client).execute(case, {"model": "fixture"})
    ground_truth = AcceptedGroundTruthRevision(
        ground_truth_revision_id="gtr", case_id=case.case_id, rubric=rubric, rubric_hash="h",
        accepted_by=ACTOR, accepted_at="2026-10-24T00:00:00+00:00",
    )
    return case, output, deterministic_checks(case, ground_truth, output)


async def test_import_scopes_the_case_to_the_document_and_renders_its_evidence_as_production(db) -> None:
    client = Client("unrelated")

    case, output, _ = await _replayed(
        db, situation("The test environment deploys every Thursday."), client, {"expected_incumbents": {}},
    )

    assert (case.source_id, case.doc_id, case.source_unit_id) == ("source-1", "doc-release", "unit-release")
    [prompt] = client.prompts
    assert "The test environment deploys every Thursday." in prompt
    assert '"observation_id":"body"' in prompt and '"revision_id":"v2"' in prompt
    assert "Production releases ship every Friday." in prompt
    assert output["outcomes"] == {"friday": "kept"}


@pytest.mark.parametrize(
    ("relation", "support", "outcome"),
    [
        # A contradiction of a claim that still holds is held for Review, not applied.
        ("contradicts", "supported", "review"),
        ("contradicts", "unsupported", "retired"),
        # DestructiveValidation keeps a removal whose read did not cover the whole revision.
        ("contradicts", {"result": "unsupported", "complete_read": False}, "kept"),
        ("refines", "supported", "revised"),
        ("unrelated", "unaffected", "kept"),
    ],
)
async def test_each_incumbent_reports_what_the_validated_operations_do_to_it(db, relation, support, outcome) -> None:
    direction = "challenger_to_candidate" if relation == "refines" else "symmetric"

    _, output, _ = await _replayed(
        db, situation("Production releases ship every Friday after two reviews.", support=support),
        Client(relation, direction), {"expected_incumbents": {}},
    )

    assert output["failure"] is None
    assert output["outcomes"] == {"friday": outcome}


async def test_a_label_gates_the_case_and_an_unlabelled_change_needs_review(db) -> None:
    _, _, labelled = await _replayed(
        db, situation("The test environment deploys every Thursday.", support="unsupported"),
        Client("contradicts"), {"expected_incumbents": {"friday": "kept"}},
    )
    _, _, unlabelled = await _replayed(
        db, situation("Production releases ship every Thursday.", support="unsupported"),
        Client("contradicts"), {"expected_incumbents": {}},
    )

    assert [(check.criterion, check.label, check.reason_code) for check in labelled[-1:]] == [
        ("incumbent_outcome", DeterministicCheckLabel.FAIL, "kept:retired"),
    ]
    assert [(check.criterion, check.label, check.reason_code) for check in unlabelled[-1:]] == [
        ("unlabelled_outcome", DeterministicCheckLabel.UNKNOWN, "retired"),
    ]


async def test_an_unknown_document_or_an_invalid_situation_is_not_imported(db) -> None:
    unknown = await _import(db, ReconciliationCaseRequest(
        input=situation("Anything."), rubric={"expected_incumbents": {}}, doc_id="missing",
    ))

    assert unknown.cases == () and unknown.skipped["document_not_found"] == ("missing",)
    with pytest.raises(ValueError, match="exactly one Primary"):
        broken = situation("Anything.")
        broken["candidates"][0]["evidence"][0]["role"] = "required"
        await _import(db, ReconciliationCaseRequest(input=broken, rubric={}, doc_id=DOCUMENT.doc_id))
    with pytest.raises(ValueError, match="names only pinned incumbents"):
        await _import(db, ReconciliationCaseRequest(
            input=situation("Anything."), rubric={"expected_incumbents": {"monday": "kept"}}, doc_id=DOCUMENT.doc_id,
        ))


async def test_the_seed_route_is_for_a_maintenance_operator_and_reports_skips(db, tmp_path) -> None:
    import httpx

    from memforge.config import AppConfig
    from memforge.server.admin_api import create_admin_app

    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    body = {
        "selection_policy_version": "reconciliation-v1",
        "cases": [{"doc_id": "doc-unsynced", "input": situation("Anything."), "rubric": {"expected_incumbents": {}}}],
    }
    responses = {}
    for operator in (False, True):
        app = create_admin_app(
            db=db, config=config, principal_resolver=lambda _request: ACTOR,
            workspace_role_resolver=lambda _request: "workspace_admin",
            maintenance_operator_resolver=lambda _request, operator=operator: operator,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            responses[operator] = await client.post("/api/v1/agent-evaluations/reconciliation-cases/seed", json=body)

    assert responses[False].status_code == 403
    assert responses[True].status_code == 200, responses[True].text
    assert responses[True].json()["skipped"]["document_not_found"] == ["doc-unsynced"]
