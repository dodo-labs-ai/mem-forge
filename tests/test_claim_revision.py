import pytest

from memforge.llm.structured import (
    ClaimRevisionDecision,
    MemoryRelationAssessment,
    RevisionAssessment,
)
from memforge.models import RawMemory, ReconcileAction
from memforge.pipeline.reconciler import SupportAuditEntry, reconcile_memories
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
            )
            if self.classification == "refines" and self.direction == "challenger_to_candidate"
            else None,
        )
        return sparse_response(prompt, [response, response] if self.duplicate else [response])


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
    [old] = catalog_payload(client.prompts[0])["existing_claims"]
    assert set(old) == {"id", "text", "type", "valid_from", "valid_until"}
    assert "current_support" not in client.prompts[0] and "evidence_status" not in client.prompts[0]


@pytest.mark.asyncio
async def test_uncertain_relation_never_becomes_add():
    client = Client("equivalent", status="insufficient")
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        doc_type="policy",
        structured_llm_client=client,
        support_audits=[SupportAuditEntry("memory", True)],
        include_metadata=True,
    )
    assert result.failure is None
    [operation] = result.operations
    assert operation.action == ReconcileAction.NOOP and operation.memory is None
    assert operation.support_revalidation_skipped
    assert client.calls == 1


@pytest.mark.asyncio
async def test_equivalent_candidate_never_keeps_an_unsupported_incumbent():
    client = Client("equivalent")
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        doc_type="policy",
        structured_llm_client=client,
        support_audits=[SupportAuditEntry("memory", False, "Current source no longer states it")],
        include_metadata=True,
    )
    assert result.failure is None
    [operation] = result.operations
    assert operation.action == ReconcileAction.DELETE and operation.memory_id == "memory"
    assert operation.flag_for_review
    assert client.calls == 1


@pytest.mark.asyncio
async def test_duplicate_edge_and_missing_candidate_fail_closed():
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
    assert (await reconcile_memories(**args)).failure is not None
    client.invalid = True
    client.calls = 0
    result = await reconcile_memories(**args)
    assert result.failure is not None and not result.operations
    # One bounded correction with the same input, then fail closed.
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
async def test_refinement_preserving_an_unsupported_incumbent_stays_unresolved():
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


@pytest.mark.asyncio
async def test_refinement_that_drops_unsupported_incumbent_truth_removes_its_support():
    client = Client("refines", "challenger_to_candidate", eligibility=False)
    result = await reconcile_memories(
        new_extractions=[candidate()],
        existing_memories=[memory()],
        doc_type="policy",
        structured_llm_client=client,
        support_audits=[SupportAuditEntry("memory", False)],
        include_metadata=True,
    )
    assert result.failure is None
    assert [op.action for op in result.operations] == [ReconcileAction.ADD, ReconcileAction.DELETE]


@pytest.mark.parametrize("stage", ["support", "admission", "claim"])
def test_semantic_assessment_contract_change_invalidates_operation_reuse(monkeypatch, stage):
    from memforge.memory import engine
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
    module, field = {
        "support": (revision_assessment, "REVISION_SUPPORT_CONTRACT"),
        "admission": (engine, "CANDIDATE_ADMISSION_CONTRACT"),
        "claim": (claim_revision, "CLAIM_REVISION_CONTRACT"),
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
async def test_revision_conditions_map_independently_to_the_lifecycle_gate(condition):
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
        doc_type="policy",
        structured_llm_client=client,
        support_audits=[SupportAuditEntry("memory", True)],
        include_metadata=True,
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
            return response.model_copy(update={"results": [response.results[0],
                response.results[0].model_copy(update={"candidate_id": "NEW-0002"})]})
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
