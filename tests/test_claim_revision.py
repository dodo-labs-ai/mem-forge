import pytest

from memforge.llm.structured import (
    ClaimRevisionResponse,
    ClaimRevisionDecision,
    MemoryRelationAssessment,
    RevisionAssessment,
)
from memforge.models import RawMemory, ReconcileAction
from memforge.pipeline.reconciler import SupportAuditEntry, reconcile_memories
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from tests.test_revision_assessment import revisions, memory


from tests.revision_client_fixture import RevisionClientFixture


class Client(RevisionClientFixture):
    def __init__(self, classification, direction="symmetric", status="resolved", eligibility=True, consistent=True):
        self.calls = 0
        self.classification, self.direction = classification, direction
        self.status, self.eligibility, self.consistent = status, eligibility, consistent
        self.duplicate = False
        self.invalid = False
        self.prompts = []

    def request_fits(self, prompt, **kwargs):
        return True

    async def assess_claim_revisions(self, prompt, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        if self.invalid:
            return ClaimRevisionResponse(decisions=[])
        response = ClaimRevisionDecision(
            pair_index=0,
            status=self.status,
            consistent_with_support=self.consistent,
            relation=MemoryRelationAssessment(
                classification=self.classification,
                direction=self.direction,
                same_subject_and_scope=self.classification == "contradicts",
                incompatible_assertions="Two versus one reviewer" if self.classification == "contradicts" else "",
            ),
            revision_assessment=RevisionAssessment(
                same_knowledge_item=True,
                preserves_incumbent_truth=self.eligibility,
                challenger_is_complete_current_claim=True,
                current_evidence_entails_challenger=True,
            )
            if self.classification == "refines" and self.direction == "challenger_to_candidate"
            else None,
        )
        return ClaimRevisionResponse(decisions=[response, response] if self.duplicate else [response])


def candidate():
    _, target = revisions("Two reviewers required.\n", "Two reviewers from distinct teams required.\n\nCountry: US.\n")
    ctx = RevisionAssessmentContext(projection=target, base=None, access_context_hash="scope")
    catalog = ctx.catalog(ctx.full_fragments)
    primary = next(f.reference for f in catalog.fragments if "distinct" in f.presentation_text)
    required = next(f.reference for f in catalog.fragments if "Country" in f.presentation_text)
    return RawMemory(
        content="US releases require two reviewers from distinct teams.",
        memory_type="fact",
        resolved_evidence_selection=catalog.resolve_selection(primary_ref=primary, required_refs=[required]),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relation,direction,supported,eligibility,actions",
    [
        ("equivalent", "symmetric", True, True, [ReconcileAction.NOOP]),
        ("refines", "challenger_to_candidate", True, True, [ReconcileAction.UPDATE]),
        ("refines", "challenger_to_candidate", True, False, [ReconcileAction.ADD, ReconcileAction.NOOP]),
        ("refines", "candidate_to_challenger", True, True, [ReconcileAction.ADD, ReconcileAction.NOOP]),
        ("contradicts", "symmetric", False, True, [ReconcileAction.SUPERSEDE]),
        ("unrelated", "symmetric", True, True, [ReconcileAction.ADD, ReconcileAction.NOOP]),
    ],
)
async def test_one_call_relation_and_revision_action_matrix(relation, direction, supported, eligibility, actions):
    client = Client(relation, direction, eligibility=eligibility)
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        doc_type="policy",
        structured_llm_client=client,
        support_audits=[SupportAuditEntry("memory", supported)],
        include_metadata=True,
    )
    assert result.failure is None
    assert [op.action for op in result.operations] == actions
    assert client.calls == 1
    assert "Country: US." in client.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,supported,consistent", [("insufficient", True, True), ("resolved", False, True)]
)
async def test_uncertainty_or_l3_inconsistency_never_becomes_add(status, supported, consistent):
    client = Client("equivalent", status=status, consistent=consistent)
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        doc_type="policy",
        structured_llm_client=client,
        support_audits=[SupportAuditEntry("memory", supported)],
        include_metadata=True,
    )
    assert result.failure is None
    [operation] = result.operations
    assert operation.action == ReconcileAction.NOOP and operation.memory is None
    assert operation.support_revalidation_skipped
    assert client.calls == 1


@pytest.mark.asyncio
async def test_duplicate_slot_is_normalized_but_missing_coverage_is_bounded():
    client = Client("equivalent")
    client.duplicate = True
    args = dict(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        doc_type="policy",
        structured_llm_client=client,
        support_audits=[SupportAuditEntry("memory", True)],
        include_metadata=True,
    )
    assert (await reconcile_memories(**args)).failure is None
    client.invalid = True
    client.calls = 0
    result = await reconcile_memories(**args)
    assert result.failure is not None and not result.operations
    assert client.calls == 2


@pytest.mark.asyncio
async def test_single_pair_does_not_require_large_model_output_window():
    client = Client("equivalent")
    requested = []

    def fits(prompt, **kwargs):
        requested.append(kwargs["max_tokens"])
        return kwargs["max_tokens"] <= 8192

    client.request_fits = fits
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        doc_type="policy",
        structured_llm_client=client,
        support_audits=[SupportAuditEntry("memory", True)],
        include_metadata=True,
    )
    assert result.failure is None and client.calls == 1
    assert max(requested) <= 8192


@pytest.mark.asyncio
async def test_refinement_entailment_chain_cannot_override_rejected_old_support():
    client = Client("refines", "challenger_to_candidate", eligibility=True)
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        doc_type="policy",
        structured_llm_client=client,
        support_audits=[SupportAuditEntry("memory", False)],
        include_metadata=True,
    )
    assert result.failure is None
    [operation] = result.operations
    assert operation.action == ReconcileAction.NOOP and operation.memory is None
    assert operation.support_revalidation_skipped
    assert client.calls == 1


@pytest.mark.parametrize("stage", ["support", "claim"])
def test_semantic_assessment_contract_change_invalidates_operation_reuse(monkeypatch, stage):
    from memforge.memory.engine import _source_lifecycle_operation_input_hash
    from memforge.pipeline import revision_assessment, claim_revision

    _, target = revisions("Two reviewers required.\n", "Two reviewers from distinct teams required.\n")
    inputs = dict(
        projection=target,
        candidates=[candidate()],
        incumbents=[memory()],
        support_hashes={"memory": "support"},
        gate_state="enabled",
        update_mode="incremental",
        changed_hunks=None,
        update_plan_stats=None,
        llm_model="test",
    )
    before = _source_lifecycle_operation_input_hash(**inputs)
    module, field = (
        (revision_assessment, "REVISION_SUPPORT_CONTRACT")
        if stage == "support"
        else (claim_revision, "CLAIM_REVISION_CONTRACT")
    )
    monkeypatch.setattr(module, field, "a-future-semantic-contract")
    assert _source_lifecycle_operation_input_hash(**inputs) != before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "condition",
    [
        "same_knowledge_item",
        "challenger_is_complete_current_claim",
        "current_evidence_entails_challenger",
    ],
)
async def test_revision_conditions_map_independently_to_the_lifecycle_gate(condition):
    client = Client("refines", "challenger_to_candidate")
    original = client.assess_claim_revisions

    async def assess(*args, **kwargs):
        response = await original(*args, **kwargs)
        proof = response.decisions[0].revision_assessment
        response.decisions[0].revision_assessment = proof.model_copy(update={condition: False})
        return response

    client.assess_claim_revisions = assess
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        doc_type="policy",
        structured_llm_client=client,
        support_audits=[SupportAuditEntry("memory", True)],
        include_metadata=True,
    )
    assert result.failure is None
    if condition == "current_evidence_entails_challenger":
        [operation] = result.operations
        assert operation.action == ReconcileAction.NOOP and operation.memory is None
        assert operation.support_revalidation_skipped
    else:
        assert [op.action for op in result.operations] == [ReconcileAction.ADD, ReconcileAction.NOOP]
    assert client.calls == 1


@pytest.mark.asyncio
async def test_large_pair_group_subdivides_without_losing_pairs():
    import json
    from dataclasses import replace
    from memforge.pipeline.claim_revision import assess_claim_pairs

    class BudgetClient(Client):
        def request_fits(self, prompt, **kwargs):
            return prompt.count('"pair_index":') <= 2

        async def assess_claim_revisions(self, prompt, **kwargs):
            self.prompts.append(prompt)
            groups = json.loads(prompt.split("<memory_pair_groups>")[1].split("</memory_pair_groups>")[0])
            return ClaimRevisionResponse(
                decisions=[
                    ClaimRevisionDecision(
                        pair_index=item["pair_index"],
                        status="resolved",
                        consistent_with_support=True,
                        relation=MemoryRelationAssessment(
                            classification="unrelated",
                            direction="symmetric",
                            same_subject_and_scope=False,
                            incompatible_assertions="",
                        ),
                    )
                    for group in groups
                    for item in group["candidates"]
                ]
            )

    client = BudgetClient("unrelated")
    olds = [replace(memory(), id=f"memory-{i}") for i in range(5)]
    result = await assess_claim_pairs(
        candidates=[candidate()],
        incumbents=olds,
        support_audits=[SupportAuditEntry(old.id, True) for old in olds],
        client=client,
        model="fixture",
    )
    assert {(index, old_id) for index, old_id, _ in result.decisions} == {(0, old.id) for old in olds}
    assert len(client.prompts) == 3


@pytest.mark.asyncio
async def test_small_output_cap_can_assess_one_complete_pair():
    from memforge.llm.request_budget import RequestBudget
    client = Client("equivalent")
    client.request_budget = lambda model=None: RequestBudget("fixture", 16000, 16000, 1024, .8, "fixture")
    client.request_fits = lambda prompt, **kwargs: kwargs["max_tokens"] <= 1024
    result = await reconcile_memories(new_extractions=[candidate()], existing_memories=[memory()],
        doc_type="policy", structured_llm_client=client, support_audits=[SupportAuditEntry("memory", True)],
        include_metadata=True)
    assert result.failure is None and client.calls == 1


@pytest.mark.asyncio
async def test_conflicting_current_refiners_skip_their_incumbent():
    from memforge.llm.structured import MemoryRelationResponse, MemoryRelationDecision
    class Conflicting(Client):
        async def assess_claim_revisions(self, prompt, **kwargs):
            response = await super().assess_claim_revisions(prompt, **kwargs)
            return response.model_copy(update={"decisions": [response.decisions[0],
                response.decisions[0].model_copy(update={"pair_index": 1})]})
        async def classify_memory_relations(self, prompt, **kwargs):
            return MemoryRelationResponse(decisions=[MemoryRelationDecision(pair_index=0, classification="contradicts",
                direction="symmetric", same_subject_and_scope=True, incompatible_assertions="Mutually exclusive refinements")])
    client = Conflicting("refines", "challenger_to_candidate")
    result = await reconcile_memories(new_extractions=[candidate(), candidate()], existing_memories=[memory()],
        doc_type="policy", structured_llm_client=client, support_audits=[SupportAuditEntry("memory", True)],
        include_metadata=True)
    assert result.failure is None
    [operation] = result.operations
    assert operation.memory is None and operation.support_revalidation_skipped
