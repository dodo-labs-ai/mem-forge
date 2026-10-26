import pytest
from pydantic import ValidationError

from memforge.llm.structured import (
    ClaimRevisionDecision,
    SameUnitRelationAssessment,
    RevisionAssessment,
    StructuredLlmError,
)
from memforge.models import RawMemory, ReconcileAction
from memforge.pipeline.reconciler import reconcile_memories
from tests.revision_client_fixture import pinned
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from tests.test_revision_assessment import revisions, memory


from tests.revision_client_fixture import RevisionClientFixture, sparse_response, catalog_payload


class Client(RevisionClientFixture):
    def __init__(self, classification, direction="symmetric", status="resolved", eligibility=True):
        self.calls = 0
        self.classification, self.direction = classification, direction
        self.status, self.eligibility = status, eligibility
        self.duplicate = False
        self.invalid = False
        self.prompts = []

    def request_fits(self, prompt, **kwargs):
        return True

    async def assess_claim_revisions(self, prompt, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        if self.invalid:
            from memforge.llm.structured import ClaimRevisionWireResponse
            return ClaimRevisionWireResponse(results=[])
        response = ClaimRevisionDecision(
            pair_index=0,
            status=self.status,
            relation=SameUnitRelationAssessment(
                classification=self.classification,
                direction=self.direction,
                same_subject_and_scope=self.classification == "contradicts",
                incompatible_assertions="Two versus one reviewer" if self.classification == "contradicts" else "",
            ),
            revision_assessment=RevisionAssessment(
                same_knowledge_item=True,
                preserves_incumbent_truth=self.eligibility,
                challenger_is_complete_current_claim=True,
            )
            if self.classification == "refines" and self.direction == "challenger_to_candidate"
            else None,
        )
        return schema_checked(lambda: sparse_response(prompt, [response, response] if self.duplicate else [response]))


def schema_checked(build):
    """Raise a response the schema rejects as the structured client does: an invalid response."""
    try:
        return build()
    except ValidationError as error:
        raise StructuredLlmError(str(error), terminal_category="invalid_response", error_code="ValidationError") from error


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
        llm_model="test-model", structured_llm_client=client,
        supports=dict([pinned("memory", supported)]),
    )
    assert result.failure is None
    assert [op.action for op in result.operations] == actions
    assert client.calls == 1
    assert "Country: US." in client.prompts[0]
    [old] = catalog_payload(client.prompts[0])["existing_claims"]
    assert set(old) == {"id", "text", "type", "valid_from", "valid_until"}
    assert "current_support" not in client.prompts[0] and "evidence_status" not in client.prompts[0]


@pytest.mark.asyncio
async def test_uncertain_relation_never_becomes_add():
    client = Client("equivalent", status="insufficient")
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        llm_model="test-model", structured_llm_client=client,
        supports=dict([pinned("memory", True)]),
    )
    assert result.failure is None
    [operation] = result.operations
    assert operation.action == ReconcileAction.NOOP and operation.memory is None
    assert operation.support_revalidation_skipped
    assert client.calls == 1


@pytest.mark.asyncio
async def test_an_unsupported_incumbent_keeps_its_result_and_its_equivalent_candidate_is_added():
    client = Client("equivalent")
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        llm_model="test-model", structured_llm_client=client,
        supports=dict([pinned("memory", False, "Current source no longer states it")]),
    )
    assert result.failure is None
    # Without a re-check the claim stays unsupported: it loses this source's Support beside the added Candidate.
    assert [recheck.memory_id for recheck in result.rechecks] == ["memory"]
    assert [(op.action, op.memory_id) for op in result.operations] == [
        (ReconcileAction.ADD, None), (ReconcileAction.DELETE, "memory"),
    ]
    assert not any(op.support_revalidation_skipped for op in result.operations)
    assert client.calls == 1


@pytest.mark.asyncio
async def test_duplicate_edge_and_missing_candidate_fail_closed():
    client = Client("equivalent")
    client.duplicate = True
    args = dict(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        llm_model="test-model",
        structured_llm_client=client,
        supports=dict([pinned("memory", True)]),
    )
    # A repeated incumbent in the Candidate's row and a missing row are both row rules: one re-ask each.
    for invalid, calls in ((False, 2), (True, 2)):
        client.invalid = invalid
        client.calls = 0
        result = await reconcile_memories(**args)
        # The Candidate alone stays invalid: it is consumed without ADD, and the old Memory is only kept.
        assert result.failure is None and result.unresolved_candidate_count == 1
        assert [(op.action, op.memory_id) for op in result.operations] == [(ReconcileAction.NOOP, "memory")]
        assert client.calls == calls


@pytest.mark.asyncio
async def test_refinement_preserving_an_unsupported_incumbent_stays_unresolved():
    client = Client("refines", "challenger_to_candidate", eligibility=True)
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        llm_model="test-model", structured_llm_client=client,
        supports=dict([pinned("memory", False)]),
    )
    assert result.failure is None
    [operation] = result.operations
    assert operation.action == ReconcileAction.NOOP and operation.memory is None
    assert operation.support_revalidation_skipped
    assert client.calls == 1


@pytest.mark.asyncio
async def test_refinement_that_drops_unsupported_incumbent_truth_removes_its_support():
    client = Client("refines", "challenger_to_candidate", eligibility=False)
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        llm_model="test-model", structured_llm_client=client,
        supports=dict([pinned("memory", False)]),
    )
    assert result.failure is None
    assert [op.action for op in result.operations] == [ReconcileAction.ADD, ReconcileAction.DELETE]


@pytest.mark.parametrize("stage", ["support", "admission", "claim", "pair_review"])
def test_semantic_assessment_contract_change_invalidates_operation_reuse(monkeypatch, stage):
    from memforge.memory import engine
    from memforge.memory.engine import _source_lifecycle_operation_input_hash
    from memforge.memory import relation_classifier
    from memforge.pipeline import revision_assessment, claim_revision

    _, target = revisions("Two reviewers required.\n", "Two reviewers from distinct teams required.\n")
    inputs = dict(
        projection=target,
        candidates=[candidate()],
        incumbents=[memory()],
        support_hashes={"memory": "support"},
        update_mode="incremental",
        changed_hunks=None,
        update_plan_stats=None,
        llm_model="test",
        decision_task_models={},
    )
    before = _source_lifecycle_operation_input_hash(**inputs)
    module, field = {
        "support": (revision_assessment, "REVISION_SUPPORT_CONTRACT"),
        "admission": (engine, "CANDIDATE_ADMISSION_CONTRACT"),
        "claim": (claim_revision, "CLAIM_REVISION_CONTRACT"),
        "pair_review": (relation_classifier, "MEMORY_PAIR_REVIEW_CONTRACT"),
    }[stage]
    monkeypatch.setattr(module, field, "a-future-semantic-contract")
    assert _source_lifecycle_operation_input_hash(**inputs) != before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "condition",
    [
        "same_knowledge_item",
        "challenger_is_complete_current_claim",
    ],
)
async def test_each_failed_revision_condition_keeps_the_old_memory(condition):
    client = Client("refines", "challenger_to_candidate")
    original = client.assess_claim_revisions

    async def assess(*args, **kwargs):
        response = await original(*args, **kwargs)
        proof = response.results[0].relations[0].revision_assessment
        response.results[0].relations[0].revision_assessment = proof.model_copy(update={condition: False})
        return response

    client.assess_claim_revisions = assess
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        llm_model="test-model", structured_llm_client=client,
        supports=dict([pinned("memory", True)]),
    )
    assert result.failure is None
    assert [op.action for op in result.operations] == [ReconcileAction.ADD, ReconcileAction.NOOP]
    assert client.calls == 1


@pytest.mark.asyncio
async def test_large_pair_group_subdivides_without_losing_pairs():
    from dataclasses import replace
    from memforge.pipeline.claim_revision import assess_claim_pairs

    class BudgetClient(Client):
        def request_fits(self, prompt, **kwargs):
            return len(catalog_payload(prompt)["existing_claims"]) <= 2

        async def assess_claim_revisions(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return sparse_response(prompt, [])

    client = BudgetClient("unrelated")
    olds = [replace(memory(), id=f"memory-{i}") for i in range(5)]
    result = await assess_claim_pairs(
        candidates=[candidate()],
        incumbents=olds,
        client=client,
        model="fixture",
    )
    assert result.decisions == ()
    assert {m["id"] for p in client.prompts for m in catalog_payload(p)["existing_claims"]} == {f"MEM-{i:04d}" for i in range(1, 6)}
    assert len(client.prompts) == 3


@pytest.mark.asyncio
async def test_chunked_incumbents_merge_relations_and_uncertainty_per_chunk():
    from dataclasses import replace
    from memforge.llm.structured import ClaimRevisionWireResponse
    from memforge.pipeline.claim_revision import assess_claim_pairs

    class ChunkClient(Client):
        def request_fits(self, prompt, **kwargs):
            return len(catalog_payload(prompt)["existing_claims"]) <= 1

        async def assess_claim_revisions(self, prompt, **kwargs):
            self.prompts.append(prompt)
            data = catalog_payload(prompt)
            [new], [old] = data["new_claims"], data["existing_claims"]
            second = old["id"] == "MEM-0002"
            return ClaimRevisionWireResponse.model_validate({"results": [{
                "candidate_id": new["id"],
                "relations": [] if second else [{"existing_id": old["id"], "relation": "equivalent", "reason": "Same rule"}],
                "uncertain_existing_ids": [old["id"]] if second else [],
            }]})

    client = ChunkClient("equivalent")
    olds = [replace(memory(), id=f"memory-{i}") for i in range(2)]
    result = await assess_claim_pairs(candidates=[candidate()], incumbents=olds, client=client, model="fixture")
    assert len(client.prompts) == 2
    assert [(incumbent, decision.status) for _, incumbent, decision in result.decisions] == [
        ("memory-0", "resolved"), ("memory-1", "insufficient"),
    ]


@pytest.mark.asyncio
async def test_small_output_cap_can_assess_one_complete_pair():
    from memforge.llm.request_budget import RequestBudget
    client = Client("equivalent")
    client.request_budget = lambda model=None: RequestBudget("fixture", 16000, 16000, 1024, .8, "fixture")
    client.request_fits = lambda prompt, *, reserve_correction=True, **kwargs: client.request_budget().fits(
        len(prompt.split()), reserve_correction=reserve_correction)
    result = await reconcile_memories(new_extractions=[candidate()], existing_memories=[memory()],
        llm_model="test-model", structured_llm_client=client, supports=dict([pinned("memory", True)]))
    assert result.failure is None and client.calls == 1


@pytest.mark.asyncio
async def test_conflicting_current_refiners_skip_their_incumbent():
    from memforge.llm.structured import SameUnitPairResponse, SameUnitPairDecision
    class Conflicting(Client):
        async def assess_claim_revisions(self, prompt, **kwargs):
            response = await super().assess_claim_revisions(prompt, **kwargs)
            return response.model_copy(update={"results": [response.results[0],
                response.results[0].model_copy(update={"candidate_id": "NEW-0002"})]})
        async def classify_same_unit_pairs(self, prompt, **kwargs):
            return SameUnitPairResponse(decisions=[SameUnitPairDecision(pair_index=0, classification="contradicts")])
    client = Conflicting("refines", "challenger_to_candidate")
    result = await reconcile_memories(new_extractions=[candidate(), candidate()], existing_memories=[memory()],
        llm_model="test-model", structured_llm_client=client, supports=dict([pinned("memory", True)]))
    assert result.failure is None
    [operation] = result.operations
    assert operation.memory is None and operation.support_revalidation_skipped


async def _refine(client, *, supported=True):
    return await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        llm_model="test-model", structured_llm_client=client,
        supports=dict([pinned("memory", supported)]),
    )


@pytest.mark.asyncio
async def test_a_revision_applies_only_when_its_pair_alone_is_verified():
    preserving, dropping = Client("refines", "challenger_to_candidate"), Client("refines", "challenger_to_candidate")
    dropping.refinement_answer = "drops"

    revised, kept = await _refine(preserving), await _refine(dropping)

    assert [op.action for op in revised.operations] == [ReconcileAction.UPDATE]
    # A dropped fact counts as an ineligible proof: the old Memory stays and the Candidate is added.
    assert [op.action for op in kept.operations] == [ReconcileAction.ADD, ReconcileAction.NOOP]
    [prompt] = preserving.verified_prompts
    assert "US releases require two reviewers." in prompt
    assert "US releases require two reviewers from distinct teams." in prompt
    assert '"pair_index": 0' in prompt and "<claim_catalog>" not in prompt


@pytest.mark.asyncio
async def test_a_revision_that_cannot_be_verified_stays_uncertain():
    from memforge.llm.structured import RefinementVerificationDecision, RefinementVerificationResponse

    class Unanswerable(Client):
        async def verify_refinements(self, prompt, **kwargs):
            return RefinementVerificationResponse(
                decisions=[RefinementVerificationDecision(pair_index=7, answer="preserves")],
            )

    result = await _refine(Unanswerable("refines", "challenger_to_candidate"))

    assert result.failure is None and result.unresolved_candidate_count == 1
    [operation] = result.operations
    assert operation.action == ReconcileAction.NOOP


@pytest.mark.asyncio
async def test_a_failed_verification_call_fails_the_revision():
    class Unreachable(Client):
        async def verify_refinements(self, prompt, **kwargs):
            raise StructuredLlmError("gateway timed out", terminal_category="deadline_exceeded", error_code="Timeout")

    result = await _refine(Unreachable("refines", "challenger_to_candidate"))

    assert result.failure is not None and result.failure.retryable and result.operations == []
    assert "refinement verification failed" in result.failure.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("relation", "direction", "supported", "eligibility"),
    [
        # The Support line rejected the old claim: the coordinator's unresolved rule decides, not a revision.
        ("refines", "challenger_to_candidate", False, True),
        # No eligible proof, so nothing would be revised.
        ("refines", "challenger_to_candidate", True, False),
        ("contradicts", "symmetric", True, True),
        ("equivalent", "symmetric", True, True),
    ],
)
async def test_only_a_revision_otherwise_certain_is_verified(relation, direction, supported, eligibility):
    client = Client(relation, direction, eligibility=eligibility)

    await _refine(client, supported=supported)

    assert getattr(client, "verified_prompts", []) == []



def _refinement_ledger(*memory_ids):
    from memforge.pipeline.reconciler import RelationLine
    from memforge.pipeline.support_relation_coordinator import RelationLedgerEntry, RevisionCompositionProof
    from memforge.memory.evidence import RelationDirection
    from memforge.memory.relation_classifier import SameUnitRelationType

    return RelationLine(
        entries=tuple(
            RelationLedgerEntry(index, memory_id, SameUnitRelationType.REFINES, RelationDirection.CHALLENGER_TO_CANDIDATE)
            for index, memory_id in enumerate(memory_ids)
        ),
        proofs=tuple(RevisionCompositionProof(index, memory_id, True, True, True) for index, memory_id in enumerate(memory_ids)),
        completed_candidate_count=len(memory_ids), incumbent_ids=frozenset(memory_ids),
    )


def _named(memory_id, claim):
    from dataclasses import replace as replaced

    return replaced(memory(), id=memory_id, content=claim)


@pytest.mark.asyncio
async def test_the_callers_coverage_verdict_decides_whether_revisions_are_verified():
    """Verification follows the caller's verdict on the Relation line's coverage, the one DestructiveValidation reads."""
    from memforge.pipeline.reconciler import join_support_and_relation

    results = {}
    for complete in (True, False):
        client = Client("refines", "challenger_to_candidate")
        client.refinement_answer = "drops"
        results[complete] = (client, await join_support_and_relation(
            _refinement_ledger("memory"), new_extractions=[candidate()], existing_memories=[memory()],
            supports=dict([pinned("memory", True)]), structured_llm_client=client, llm_model="test-model",
            relation_complete=complete,
        ))

    verifying, verified = results[True]
    assert len(verifying.verified_prompts) == 1
    assert ReconcileAction.UPDATE not in [op.action for op in verified.operations]
    # An incomplete line withholds every revision downstream, so nothing is asked.
    skipping, _ = results[False]
    assert getattr(skipping, "verified_prompts", []) == []


@pytest.mark.asyncio
async def test_each_pair_takes_its_own_answer():
    from memforge.llm.structured import RefinementVerificationDecision, RefinementVerificationResponse
    from memforge.pipeline.reconciler import join_support_and_relation

    class PerClaim(Client):
        async def verify_refinements(self, prompt, **kwargs):
            answer = "drops" if "Old claim one." in prompt else "preserves"
            index = 7 if "Old claim three." in prompt else 0
            return RefinementVerificationResponse(decisions=[RefinementVerificationDecision(pair_index=index, answer=answer)])

    incumbents = [_named("m1", "Old claim one."), _named("m2", "Old claim two."), _named("m3", "Old claim three.")]
    candidates = [RawMemory(content=f"New claim {n}.", memory_type="fact") for n in ("one", "two", "three")]

    result = await join_support_and_relation(
        _refinement_ledger("m1", "m2", "m3"), new_extractions=candidates, existing_memories=incumbents,
        supports={memory_id: pinned(memory_id, True)[1] for memory_id in ("m1", "m2", "m3")},
        structured_llm_client=PerClaim("refines", "challenger_to_candidate"), llm_model="test-model",
        relation_complete=True,
    )

    by_memory = {op.memory_id: op for op in result.operations if op.memory_id}
    assert by_memory["m1"].action is ReconcileAction.NOOP and by_memory["m3"].action is ReconcileAction.NOOP
    assert by_memory["m2"].action is ReconcileAction.UPDATE
    # The dropped revision adds its Candidate beside m1; the unjudged one is consumed and leaves m3 unresolved.
    assert result.unresolved_candidate_count == 1
    assert [op.memory.content for op in result.operations if op.action is ReconcileAction.ADD] == ["New claim one."]
    assert (result.metrics.refinement_verification_count, result.metrics.refinement_verification_dropped_count,
            result.metrics.refinement_verification_unjudged_count) == (3, 1, 1)
