"""Support Assessment cases: pinned from the receipts of a real Support read, then replayed.

The fixture client stands in for the model; this is contract coverage, not
model-accuracy evidence.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.evals.offline_evaluation import (
    AcceptedGroundTruthRevision,
    AgentEvaluationCaseKind,
    AgentEvaluationRole,
    DeterministicCheckLabel,
    OfflineAgentEvaluation,
    PinnedSupportInput,
    SourceUnitSupportReplayExecutor,
    deterministic_checks,
    rubric_memory_labels,
)
from memforge.evals.source_unit_support_cases import SupportCaseRequest, seed_source_unit_support_cases
from memforge.derivation_work import DerivationWork
from memforge.memory.evidence import EvidencePartKind, EvidenceReference
from memforge.models import DocumentRecord
from memforge.pipeline.revision_work import RevisionWorkExecutor, SupportRecheck
from memforge.pipeline.support_work import plan_support_work
from memforge.source_derivation import SourceDerivationAttempt, SourceUnitDerivationContext
from memforge.storage.database import Database
from tests.test_revision_assessment import memory, old_support, revisions
from tests.test_revision_work import CHANGED, RULE, Client, Store

ACTOR = "dev"
DOC_ID = "doc"
CANDIDATE = {"model": "fixture"}


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "support-cases.db"))
    await database.connect()
    await database.upsert_source(
        id="source-1", type="github_repo", name="Support cases", config_json="{}",
        access_policy="workspace", owner_user_id="fixture-user",
    )
    yield database
    await database.close()


class Synced:
    """One applied update whose Supports the fixture model read, with the receipts it left."""

    def __init__(self, target_text=CHANGED, *, memories=("memory-0", "memory-1")):
        self.base, self.target = revisions(f"{RULE}\n", target_text)
        base_unit = self.base.source_unit_revisions[0].id
        self.incumbents = [replace(memory(), id=memory_id) for memory_id in memories]
        self.evidence = {
            item.id: tuple(
                replace(part, memory_id=item.id, evidence_unit_id=f"eu-{item.id}", reference_id=f"eref-{item.id}",
                        validation_plan_id="plan-v1", validation_unit_revision_id=base_unit)
                for part in old_support(self.base)
            )
            for item in self.incumbents
        }
        self.journal = Store()
        self.attempt = SourceDerivationAttempt(
            id="deriv-v2", source_id="source-1", source_unit_id=self.target.source_units[0].id,
            base_unit_revision_id=base_unit, target_unit_revision_id=self.target.source_unit_revisions[0].id,
            projection=self.target, projection_payload_hash="p", projection_identity_hash="p",
            context=SourceUnitDerivationContext(
                document=DocumentRecord(
                    doc_id=DOC_ID, source="source-1", source_url="https://example.test/doc", title="Release rules",
                    space_or_project="TEST", author=None, last_modified=datetime(2026, 9, 7, tzinfo=timezone.utc),
                    labels=[], version="2", content_hash="hash", token_count=20,
                    last_synced=datetime(2026, 9, 7, tzinfo=timezone.utc),
                ),
                doc_type="document", project_key=None, repo_identifier=None,
                document_content=self.target.observation_revisions[0].content, update_mode="update",
                changed_hunks=None, update_plan_stats=None, source_updated_at="2026-09-07T00:00:00+00:00",
                user_id=None,
            ),
            context_payload_hash="c", context_identity_hash="c", extraction_contract_version="v",
            status="applied", terminal_reason_code=None, authority_plan_identity={"access_context_hash": "scope"},
            batches=(), created_at="2026-09-07T00:00:00+00:00", updated_at="2026-09-07T00:00:00+00:00",
            completed_at="2026-09-07T00:00:00+00:00",
        )
        self.attempts = [self.attempt]
        self.revisions = {self.base.source_unit_revisions[0].id: self.base}
        self.withheld_evidence: set[str] = set()

    async def run(self, client) -> dict:
        """Production: plan the work over the store and assess it with a durable journal."""
        work = await plan_support_work(
            self, projection=self.target, base=self.base, incumbents=self.incumbents,
            evidence_by_memory=self.evidence,
            unit_support={item.id: {f"eu-{item.id}"} for item in self.incumbents}, access_context_hash="scope",
        )
        self.work = work
        self.executor = RevisionWorkExecutor(
            client=client, model="fixture", store=self.journal, derivation_id=self.attempt.id,
        )
        return await self.executor.assess_many(list(work.items))

    async def recheck(self) -> None:
        """The coordinator's normal-order re-check of every Support the first pass rebound."""
        await self.executor.recheck([SupportRecheck(item) for item in self.work.items])

    # The store a seeder reads.
    async def get_document(self, doc_id):
        return self.attempt.context.document if doc_id == DOC_ID else None

    async def list_documents(self, source=None, search=None, limit=100):
        return [self.attempt.context.document]

    async def find_source_unit_by_document_id(self, source_id, document_id, *, current_only=False):
        return self.target.source_units[0] if document_id == DOC_ID else None

    async def list_source_derivation_attempts(self, *, source_id, statuses=None, source_unit_id=None):
        return tuple(self.attempts)

    async def get_source_unit_revision_projection(self, source_unit_id, revision_id):
        return self.revisions.get(revision_id)

    async def get_source_observation_revisions(self, revision_ids):
        known = {revision.id: revision for revision in self.base.observation_revisions}
        return {revision_id: known[revision_id] for revision_id in revision_ids if revision_id in known}

    async def list_source_derivation_work(self, derivation_id):
        return tuple(work for (owner, _), work in sorted(self.journal.works.items()) if owner == derivation_id)

    async def get_evidence_references(self, reference_ids):
        parts = {
            part.reference_id: part
            for memory_id, memory_parts in self.evidence.items() if memory_id not in self.withheld_evidence
            for part in memory_parts
        }
        return {
            reference_id: EvidenceReference(
                id=reference_id, evidence_unit_id=part.evidence_unit_id, role=part.role, anchor=part.anchor,
                kind=EvidencePartKind.TEXT, raw_content_sha256=part.raw_content_sha256,
                presentation_sha256=part.presentation_sha256, excerpt=part.excerpt,
            )
            for reference_id, part in parts.items() if reference_id in reference_ids
        }


async def _seed(db, synced, *, rubric=None, derivation_id=None):
    return await seed_source_unit_support_cases(
        synced, OfflineAgentEvaluation(db, executors={}),
        requests=[SupportCaseRequest(doc_id=DOC_ID, derivation_id=derivation_id,
                                     rubric=rubric or {"expected_support": {}})],
        role=AgentEvaluationRole.SENTINEL, selection_policy_version="support-v1", actor=ACTOR,
    )


def _ground_truth(case, rubric) -> AcceptedGroundTruthRevision:
    return AcceptedGroundTruthRevision(
        ground_truth_revision_id="gtr", case_id=case.case_id, rubric=rubric, rubric_hash="h",
        accepted_by=ACTOR, accepted_at="2026-09-07T00:00:00+00:00",
    )


async def test_an_update_is_pinned_from_its_receipts_and_replayed_through_the_same_planner(db) -> None:
    synced = Synced()
    production = await synced.run(Client(limit=16000))
    assert all(assessment.supported for assessment in production.values())

    report = await _seed(db, synced)
    again = await _seed(db, synced)

    [pinned] = report.cases
    assert (pinned.doc_id, pinned.derivation_id) == (DOC_ID, "deriv-v2")
    assert again.cohort_id == report.cohort_id
    case = await db.get_agent_evaluation_case(pinned.case_id)
    assert case.case_kind is AgentEvaluationCaseKind.SOURCE_UNIT_SUPPORT
    pinned_input = PinnedSupportInput.from_manifest(case.manifest)
    # The claims are the ones the model saw, the target is the derivation's own, and
    # the recorded results come with them.
    assert [(item.recorded_work_id, item.memory.id, item.observed) for item in pinned_input.work_items] == [
        ("w000000", "memory-0", "supported"), ("w000001", "memory-1", "supported"),
    ]
    assert pinned_input.projection == synced.target
    assert pinned_input.base == synced.base
    assert pinned_input.work_items[0].support == synced.evidence["memory-0"]

    replay_client = Client(limit=16000)
    output = await SourceUnitSupportReplayExecutor(replay_client, object()).execute(case, CANDIDATE)

    assert output["supports"] == [
        {"memory_id": "memory-0", "evidence_unit_id": "eu-memory-0", "result": "supported", "complete_read": False},
        {"memory_id": "memory-1", "evidence_unit_id": "eu-memory-1", "result": "supported", "complete_read": False},
    ]
    # Every Support here reaches the model, so replay sends exactly what production sent.
    production_client = Client(limit=16000)
    await Synced().run(production_client)
    assert replay_client.prompts == production_client.prompts
    assert replay_client.impact_prompts == production_client.impact_prompts


async def test_labels_gate_the_case_and_an_unlabelled_retirement_needs_review(db) -> None:
    synced = Synced()
    await synced.run(Client(limit=16000))
    [pinned] = (await _seed(db, synced)).cases
    case = await db.get_agent_evaluation_case(pinned.case_id)
    output = {
        "supports": [
            {"memory_id": "memory-0", "evidence_unit_id": "eu-memory-0", "result": "supported"},
            {"memory_id": "memory-1", "evidence_unit_id": "eu-memory-1", "result": "unsupported"},
        ],
    }

    labelled = deterministic_checks(
        case, _ground_truth(case, {"expected_support": {"memory-0": "unsupported", "memory-1": "unsupported"}}),
        output,
    )
    unlabelled = deterministic_checks(case, _ground_truth(case, {"expected_support": {}}), output)
    unresolved = deterministic_checks(
        case, _ground_truth(case, {"expected_support": {"memory-0": "supported"}}),
        {"supports": [{"memory_id": "memory-0", "evidence_unit_id": "eu-memory-0", "result": "unresolved_capacity"}]},
    )

    assert [(check.criterion, check.label, check.reason_code, check.item_id) for check in labelled[1:]] == [
        ("support_label", DeterministicCheckLabel.FAIL, "unsupported:supported", "memory-0:eu-memory-0"),
        ("support_label", DeterministicCheckLabel.PASS, "unsupported:unsupported", "memory-1:eu-memory-1"),
    ]
    assert [(check.criterion, check.label, check.item_id) for check in unlabelled[1:]] == [
        ("unlabelled_unsupported", DeterministicCheckLabel.UNKNOWN, "memory-1:eu-memory-1"),
    ]
    assert unresolved[1].label is DeterministicCheckLabel.UNKNOWN
    assert deterministic_checks(case, _ground_truth(case, {"expected_support": {}}), {})[0].label is (
        DeterministicCheckLabel.FAIL
    )


async def test_a_rubric_names_only_pinned_memories_with_a_known_label(db) -> None:
    synced = Synced()
    await synced.run(Client(limit=16000))

    with pytest.raises(ValueError, match="only pinned Memories"):
        await _seed(db, synced, rubric={"expected_support": {"memory-9": "supported"}})
    with pytest.raises(ValueError, match="labels must be one of"):
        await _seed(db, synced, rubric={"expected_support": {"memory-0": "kept"}})


async def test_a_term_labels_every_pinned_memory_whose_claim_holds_it(db) -> None:
    synced = Synced()
    await synced.run(Client(limit=16000))
    [pinned] = (await _seed(db, synced, rubric={"expected_support_terms": {"Reviewers": "supported"}})).cases
    case = await db.get_agent_evaluation_case(pinned.case_id)
    output = {
        "supports": [
            {"memory_id": "memory-0", "evidence_unit_id": "eu-memory-0", "result": "supported"},
            {"memory_id": "memory-1", "evidence_unit_id": "eu-memory-1", "result": "unsupported"},
        ],
    }

    checks = deterministic_checks(
        case, _ground_truth(case, {"expected_support_terms": {"reviewers": "supported", "EU": "unsupported"}}), output,
    )

    assert [(check.criterion, check.label, check.reason_code, check.item_id) for check in checks[1:]] == [
        ("support_label", DeterministicCheckLabel.UNKNOWN, "term_unmatched", "term:EU"),
        ("support_label", DeterministicCheckLabel.PASS, "supported:supported", "memory-0:eu-memory-0"),
        ("support_label", DeterministicCheckLabel.FAIL, "supported:unsupported", "memory-1:eu-memory-1"),
    ]
    with pytest.raises(ValueError, match="labelled both"):
        await _seed(db, synced, rubric={
            "expected_support": {"memory-0": "unsupported"}, "expected_support_terms": {"releases": "supported"},
        })


def test_a_term_matches_a_whole_word_and_inside_unspaced_script() -> None:
    claims = {
        "api": "The API rate limit is 100.",
        "rapid": "Rapid releases ship weekly.",
        "retry": "失败任务重试三次后进入 kestrel-dlq。",
    }

    labels, unmatched = rubric_memory_labels(
        {}, {"api": "supported", "重试": "unsupported", "kestrel-dlq": "unsupported", "dlq-x": "supported"}, claims,
    )

    assert labels == {"api": "supported", "retry": "unsupported"}
    assert unmatched == ("dlq-x",)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("no_attempts", "no_update"),
        ("first_derivation", "no_update"),
        ("no_receipts", "no_support_work"),
        ("named_other", "derivation_not_found"),
        ("base_gone", "revision_unavailable"),
        ("evidence_gone", "evidence_unavailable"),
        ("other_access_context", "catalog_mismatch"),
    ],
)
async def test_an_update_that_cannot_be_pinned_exactly_is_skipped_with_its_reason(db, change, reason) -> None:
    synced = Synced()
    if change != "no_receipts":
        await synced.run(Client(limit=16000))
    derivation_id = None
    if change == "no_attempts":
        synced.attempts = []
    elif change == "first_derivation":
        synced.attempts = [replace(synced.attempt, base_unit_revision_id=None)]
    elif change == "named_other":
        derivation_id = "deriv-other"
    elif change == "base_gone":
        synced.revisions = {}
    elif change == "evidence_gone":
        synced.withheld_evidence = {"memory-1"}
    elif change == "other_access_context":
        synced.attempts = [replace(synced.attempt, authority_plan_identity={"access_context_hash": "other"})]

    report = await _seed(db, synced, derivation_id=derivation_id)

    assert (report.cohort_id, report.cases) == (None, ())
    assert report.skipped[reason] == (DOC_ID,)


async def test_the_seed_route_is_for_a_maintenance_operator_and_reports_skips(db, tmp_path) -> None:
    import httpx

    from memforge.config import AppConfig
    from memforge.server.admin_api import create_admin_app

    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    body = {"selection_policy_version": "support-v1", "cases": [{"doc_id": DOC_ID, "rubric": {"expected_support": {}}}]}
    responses = {}
    for operator in (False, True):
        app = create_admin_app(
            db=db, config=config, principal_resolver=lambda _request: ACTOR,
            workspace_role_resolver=lambda _request: "workspace_admin",
            maintenance_operator_resolver=lambda _request, operator=operator: operator,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            responses[operator] = await client.post("/api/v1/agent-evaluations/support-cases/seed", json=body)

    assert responses[False].status_code == 403
    assert responses[True].status_code == 200, responses[True].text
    assert responses[True].json()["skipped"]["document_not_found"] == [DOC_ID]


class UnaffectedClient(Client):
    def impact(self, work, data):
        return "unaffected"


async def test_a_coordinator_recheck_is_not_taken_for_the_first_pass(db) -> None:
    synced = Synced()
    first_pass = await synced.run(UnaffectedClient(limit=16000))
    assert all(assessment.rebound for assessment in first_pass.values())
    await synced.recheck()
    assert any(work.kind == "support_assess" for work in synced.journal.works.values())

    [pinned] = (await _seed(db, synced)).cases
    case = await db.get_agent_evaluation_case(pinned.case_id)

    # Change Impact judged both in the first pass; the later reads were re-checks.
    assert [item.observed for item in PinnedSupportInput.from_manifest(case.manifest).work_items] == [
        "unaffected", "unaffected",
    ]


async def test_receipts_of_another_contract_are_not_pinned(db) -> None:
    synced = Synced()
    await synced.run(Client(limit=16000))
    for key, work in list(synced.journal.works.items()):
        if work.kind == "support_assess":
            scope = {**work.manifest["scope"], "contract": "support-ordered-reading-v1"}
            synced.journal.works[key] = DerivationWork.create(work.kind, {**work.manifest, "scope": scope})

    report = await _seed(db, synced)

    assert report.skipped["contract_changed"] == (DOC_ID,)


async def test_a_replay_that_reads_images_differently_needs_review(db) -> None:
    synced = Synced()
    await synced.run(Client(limit=16000))
    [pinned] = (await _seed(db, synced)).cases
    case = await db.get_agent_evaluation_case(pinned.case_id)
    assert case.manifest["reads_images"] is None
    output = {
        "reads_images": False,
        "supports": [{"memory_id": "memory-0", "evidence_unit_id": "eu-memory-0", "result": "supported"}],
    }

    unpinned = deterministic_checks(case, _ground_truth(case, {"expected_support": {}}), output)
    pinned_reading = deterministic_checks(
        replace(case, manifest={**case.manifest, "reads_images": True}),
        _ground_truth(case, {"expected_support": {}}), output,
    )

    assert [check.criterion for check in unpinned] == ["typed_output"]
    assert [(check.criterion, check.label) for check in pinned_reading[1:]] == [
        ("image_reading", DeterministicCheckLabel.UNKNOWN),
    ]


async def test_a_program_rebind_read_only_by_a_recheck_is_not_pinned(db) -> None:
    # The new revision changed no content, so the first pass rebinds without a model call.
    synced = Synced(target_text=f"{RULE}\n")
    first_pass = await synced.run(Client(limit=16000))
    assert all(assessment.rebound for assessment in first_pass.values())
    await synced.recheck()
    assert any(work.kind == "support_assess" for work in synced.journal.works.values())

    report = await _seed(db, synced)

    assert report.skipped["no_support_work"] == (DOC_ID,)


async def test_evidence_references_read_keeps_every_stored_column_as_stored(db) -> None:
    # A migrated reference without digests or excerpt, written directly as the migration left it.
    await db.db.execute("PRAGMA foreign_keys = OFF")
    await db.db.execute(
        """INSERT INTO evidence_references (id, evidence_unit_id, role, part_kind, anchor_kind, observation_id,
               observation_revision_id, range_start, range_end, created_at)
           VALUES ('eref-1', 'eu-1', 'primary', NULL, 'revision_range', 'obs-1', 'obsrev-1', 3, 9, 'now')"""
    )
    await db.db.commit()
    await db.db.execute("PRAGMA foreign_keys = ON")

    references = await db.get_evidence_references({"eref-1", "eref-2"})

    [reference] = references.values()
    assert (reference.id, reference.evidence_unit_id, reference.kind, reference.excerpt) == (
        "eref-1", "eu-1", None, None,
    )
    assert (reference.raw_content_sha256, reference.presentation_sha256) == (None, None)
    assert (reference.anchor.observation_revision_id, reference.anchor.range_start) == ("obsrev-1", 3)
