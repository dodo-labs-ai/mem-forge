"""Sparse catalog contract: explicit edges, omission, and candidate completeness."""
from dataclasses import replace

import pytest

from memforge.llm.structured import ClaimRevisionWireResponse
from memforge.models import ReconcileAction
from memforge.pipeline.claim_revision import assess_claim_pairs
from memforge.pipeline.reconciler import SupportAuditEntry, reconcile_memories
from tests.test_claim_revision import Client, candidate, memory
from tests.revision_client_fixture import catalog_payload, sparse_response


class SparseClient(Client):
    def __init__(self, result=None):
        super().__init__("unrelated")
        self.result = result

    async def assess_claim_revisions(self, prompt, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        return ClaimRevisionWireResponse.model_validate(self.result) if self.result is not None else sparse_response(prompt, [])


@pytest.mark.asyncio
async def test_eight_by_183_uses_one_catalog_and_eight_empty_rows():
    client = SparseClient()
    olds = [replace(memory(), id=f"old-{i}", content=f"existing claim {i}") for i in range(183)]
    result = await assess_claim_pairs(candidates=[replace(candidate(), content=f"new claim {i}") for i in range(8)],
        incumbents=olds, support_audits=[SupportAuditEntry(m.id, True) for m in olds], client=client, model="fixture")
    assert client.calls == 1 and result.decisions == ()
    payload = catalog_payload(client.prompts[0])
    assert len(payload["existing_claims"]) == 183
    assert len(payload["new_claims"]) == 8
    # Each fixture candidate shares its same Primary + Required parts.
    assert len(payload["evidence_catalog"]) == 2
    assert all(set(c["evidence_refs"]) == set(payload["evidence_catalog"]) for c in payload["new_claims"])
    wire = sparse_response(client.prompts[0], [])
    assert len(wire.results) == 8
    assert len(wire.model_dump_json()) < 1500
    assert 'pair_index' not in client.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [("entailed", [ReconcileAction.ADD, ReconcileAction.NOOP]),
                                            ("insufficient", [ReconcileAction.NOOP])])
async def test_empty_relationships_and_insufficient_evidence_are_distinct(status, expected):
    client = SparseClient(dict(results=[dict(candidate_id="C0", evidence_status=status,
        relations=[], uncertain_existing_ids=[])]))
    result = await reconcile_memories(new_extractions=[candidate()], existing_memories=[memory()],
        doc_type="policy", structured_llm_client=client, support_audits=[SupportAuditEntry("memory", True)], include_metadata=True)
    assert result.failure is None
    assert [op.action for op in result.operations] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("row", [
    dict(candidate_id="C9", evidence_status="entailed", relations=[], uncertain_existing_ids=[]),
    dict(candidate_id="C0", evidence_status="entailed", relations=[], uncertain_existing_ids=["M9"]),
    dict(candidate_id="C0", evidence_status="entailed", relations=[dict(existing_id="M0", relation="equivalent")], uncertain_existing_ids=["M0"]),
])
async def test_invalid_references_and_duplicates_never_apply(row):
    client = SparseClient(dict(results=[row]))
    result = await reconcile_memories(new_extractions=[candidate()], existing_memories=[memory()],
        doc_type="policy", structured_llm_client=client, support_audits=[SupportAuditEntry("memory", True)], include_metadata=True)
    assert result.failure is not None and not result.operations


@pytest.mark.asyncio
async def test_omission_does_not_skip_unsupported_incumbent_audit():
    result = await reconcile_memories(new_extractions=[candidate()], existing_memories=[memory()],
        doc_type="policy", structured_llm_client=SparseClient(), support_audits=[SupportAuditEntry("memory", False)], include_metadata=True)
    assert result.failure is None
    assert [op.action for op in result.operations] == [ReconcileAction.ADD, ReconcileAction.DELETE]


@pytest.mark.asyncio
async def test_explicit_uncertainty_preserves_incumbent_and_consumes_candidate():
    client = SparseClient(dict(results=[dict(candidate_id="C0", evidence_status="entailed", relations=[], uncertain_existing_ids=["M0"])]))
    result = await reconcile_memories(new_extractions=[candidate()], existing_memories=[memory()],
        doc_type="policy", structured_llm_client=client, support_audits=[SupportAuditEntry("memory", False)], include_metadata=True)
    assert result.failure is None
    assert len(result.operations) == 1 and result.operations[0].support_revalidation_skipped


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["length", "max_tokens", "content_filter"])
async def test_valid_json_with_incomplete_finish_is_not_empty_success(monkeypatch, finish):
    from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig, StructuredLlmError
    from tests.test_structured_llm import CompletionResponse
    async def provider(**kwargs):
        response = CompletionResponse('{"results":[]}')
        response.choices[0].finish_reason = finish
        return response
    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", provider)
    client = LiteLlmStructuredClient(StructuredLlmConfig(model="openai/gpt-4o-mini", api_key="fixture", base_url=None,
        timeout_s=5, num_retries=0, native_schema_transport="response_format"))
    with pytest.raises(StructuredLlmError) as caught:
        await client.assess_claim_revisions("fixture", max_tokens=512)
    assert caught.value.error_code == "claim_response_incomplete"


@pytest.mark.asyncio
@pytest.mark.parametrize("omit_second", [False, True])
async def test_competing_replacements_expose_the_accepted_omission_tradeoff(omit_second):
    edges = [dict(existing_id="M0", relation="contradicts", contradiction=dict(
        same_subject_and_scope=True, incompatible_assertions="ten/fifteen minutes versus five minutes"))]
    client = SparseClient(dict(results=[
        dict(candidate_id="C0", evidence_status="entailed", relations=edges, uncertain_existing_ids=[]),
        dict(candidate_id="C1", evidence_status="entailed", relations=[] if omit_second else edges, uncertain_existing_ids=[]),
    ]))
    result = await reconcile_memories(new_extractions=[candidate(), candidate()], existing_memories=[memory()],
        doc_type="policy", structured_llm_client=client, support_audits=[SupportAuditEntry("memory", True)], include_metadata=True)
    assert result.failure is None
    assert client.calls == 1
    replacements = [op for op in result.operations if op.action == ReconcileAction.SUPERSEDE]
    if omit_second:
        assert len(replacements) == 1 and replacements[0].flag_for_review
    else:
        assert replacements == []


@pytest.mark.asyncio
async def test_candidate_block_from_one_capacity_partition_applies_to_all_its_edges():
    class Partitioned(SparseClient):
        def request_fits(self, prompt, **kwargs):
            return len(catalog_payload(prompt)["existing_claims"]) == 1
        async def assess_claim_revisions(self, prompt, **kwargs):
            ref = catalog_payload(prompt)["existing_claims"][0]["id"]
            return ClaimRevisionWireResponse.model_validate(dict(results=[dict(candidate_id="C0",
                evidence_status="insufficient" if ref == "M1" else "entailed",
                relations=[] if ref == "M1" else [dict(existing_id=ref, relation="equivalent")], uncertain_existing_ids=[])]))
    olds = [memory(), replace(memory(), id="second")]
    result = await reconcile_memories(new_extractions=[candidate()], existing_memories=olds,
        doc_type="policy", structured_llm_client=Partitioned(), support_audits=[SupportAuditEntry(m.id, True) for m in olds], include_metadata=True)
    assert result.failure is None
    assert all(op.action == ReconcileAction.NOOP and op.memory is None for op in result.operations)
    assert result.operations[0].support_revalidation_skipped
