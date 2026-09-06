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


class Client:
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
                same_memory_identity=True,
                preserves_incumbent_truth=self.eligibility,
                candidate_is_canonical_composite=True,
                current_evidence_entails_candidate=True,
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
    "status,supported,consistent", [("insufficient", True, True), ("resolved", False, True), ("resolved", True, False)]
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
    assert result.failure is not None and not result.operations
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
