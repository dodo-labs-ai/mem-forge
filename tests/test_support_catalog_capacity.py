"""Model-wire and real request-budget boundaries; no provider or source writes."""

import json
from types import SimpleNamespace

import pytest

from memforge.llm.request_budget import RequestBudget
from memforge.llm.structured import (
    OUTPUT_TRUNCATED, LiteLlmStructuredClient, StructuredLlmConfig, StructuredLlmError,
    SupportAssessmentWireResponse,
)
from memforge.pipeline.revision_work import RevisionWorkExecutor
from memforge.pipeline.support_wire import SupportWireAliases
from tests.test_revision_work import CHANGED, Client, Store, payload, work_items


class BudgetClient(Client):
    def request_budget(self, model=None):
        return RequestBudget('gpt-4o', 1000000, 1000000, 64000, .8, 'fixture')

    def request_fits(self, prompt, *, max_tokens, reserve_correction=True, **kwargs):
        return self.request_budget().fits(self.request_tokens(prompt), max_tokens, reserve_correction=reserve_correction)


@pytest.mark.asyncio
async def test_ordered_reading_fits_one_request_despite_dense_output_estimate():
    items = work_items('Two reviewers approve US releases.\n\n' + '\n\n'.join(
        f'Routine unchanged rule {i}.' for i in range(169)), 183)
    client, store = BudgetClient(), Store()
    executor = RevisionWorkExecutor(client=client, model='gpt-4o', store=store, derivation_id='root')
    results = await executor.assess_many(items)
    assert len(client.prompts) == 1
    request = payload(client.prompts[0])
    assert len(request['works']) == 183
    assert all(work['may_conclude'] for work in request['works'])
    assert len(results) == 183 and all(row.supported for row in results.values())
    work = next(w for w in store.works.values() if w.kind == 'support_assess')
    assert work.manifest['output'] == 64000
    assert work.manifest['scope']['contract'] == 'support-ordered-reading-v1'
    # The journal keeps the provider's aliased wire response; reuse decodes it again.
    assert all(row['work_id'].startswith('WRK-') for row in work.result['results'])
    assert all(row['primary_ref'].startswith('PRM-') for row in work.result['results'])
    assert all(row['work_id'].startswith('WRK-') for row in request['works'])


@pytest.mark.asyncio
async def test_truncated_multi_claim_output_is_halved_until_each_request_completes():
    class TruncatingClient(BudgetClient):
        async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
            if response_format is SupportAssessmentWireResponse and len(payload(prompt)['works']) > 1:
                self.prompts.append(prompt)
                raise StructuredLlmError('fixture truncation', error_code=OUTPUT_TRUNCATED)
            return await super().evaluate_revision_work(prompt, response_format=response_format, **kwargs)
    client = TruncatingClient()
    executor = RevisionWorkExecutor(client=client, model='gpt-4o')
    results = await executor.assess_many(work_items(CHANGED, 4))
    assert all(row.supported for row in results.values())
    assert [len(payload(p)['works']) for p in client.prompts] == [4, 2, 1, 1, 2, 1, 1]
    # One Change Impact request judged all four AFFECTED before the seven reading requests.
    assert len(client.impact_prompts) == 1 and executor.calls == 8 and len(executor.final_work_ids) == 4


def test_aliases_preserve_text_and_role_eligibility_and_expand_four_digits():
    catalog = SimpleNamespace(fragments=[
        SimpleNamespace(reference='p000068', primary_eligible=True),
        SimpleNamespace(reference='r010000', primary_eligible=False),
    ])
    wire = SupportWireAliases(catalog, [], {'canonical-task': 'WRK-10000'})
    row = {'work_id': 'WRK-10000', 'status': 'supported', 'primary_ref': 'PRM-0068',
           'required_refs': ['REQ-10000'], 'omitted_matched_refs': ['PRM-0068']}
    decoded = wire.decode(SupportAssessmentWireResponse.model_validate({'results': [row]})).results[0]
    assert decoded.work_id == 'canonical-task' and decoded.primary_ref == 'p000068'
    assert decoded.required_refs == ['r010000'] and decoded.omitted_matched_refs == ['p000068']
    # A Primary-eligible ref can also be selected as Required.
    row['required_refs'] = ['PRM-0068']
    assert wire.decode(SupportAssessmentWireResponse.model_validate({'results': [row]})).results[0].required_refs == [
        'p000068'
    ]
    row['primary_ref'] = 'REQ-10000'
    with pytest.raises(Exception, match='REQ-10000') as exc:
        wire.decode(SupportAssessmentWireResponse.model_validate({'results': [row]}))
    assert 'r010000' not in str(exc.value)
    # Historical and unknown refs are never current Evidence.
    witness = {'work_id': 'WRK-10000', 'status': 'continue',
               'witness_delta': {'support_witness_refs': ['HIS-0000'], 'opposing_witness_refs': []}}
    with pytest.raises(Exception, match='HIS-0000'):
        wire.decode(SupportAssessmentWireResponse.model_validate({'results': [witness]}))


@pytest.mark.asyncio
async def test_wire_id_text_is_not_rewritten_and_unknown_namespace_fails_closed():
    class CanonicalIDClient(Client):
        def judge(self, prompt):
            results = super().judge(prompt)
            results[0]['primary_ref'] = 'p000001'
            return results
    items = work_items('Two reviewers approve US releases.\n\nLiteral p000001 and w000068 are source text.')
    client, store = CanonicalIDClient(limit=50000), Store()
    executor = RevisionWorkExecutor(client=client, model='gpt-4o', store=store, derivation_id='root')
    with pytest.raises(Exception, match='bounded assessment correction exhausted'):
        await executor.assess_many(items)
    assert len(client.prompts) == 2
    assert 'Literal p000001 and w000068 are source text.' in client.prompts[0]
    assert all(w.status != 'completed' for w in store.works.values() if w.kind == 'support_assess')


@pytest.mark.asyncio
@pytest.mark.parametrize('signal', ['length', 'max_tokens', 'refusal', 'stop_max_tokens'])
async def test_valid_support_json_with_incomplete_transport_is_rejected(monkeypatch, signal):
    from tests.test_structured_llm import CompletionResponse, set_native_schema_support
    response = CompletionResponse(json.dumps({'results': [{
        'work_id': 'WRK-0000', 'status': 'supported', 'primary_ref': 'PRM-0001',
        'required_refs': [], 'omitted_matched_refs': [],
    }]}))
    if signal == 'refusal':
        response.choices[0].message.refusal = 'refused'
    elif signal == 'stop_max_tokens':
        response.choices[0].stop_reason = 'max_tokens'
    else:
        response.choices[0].finish_reason = signal
    async def completion(**kwargs):
        return response
    monkeypatch.setattr('memforge.llm.structured.litellm.acompletion', completion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(StructuredLlmConfig(
        model='gpt-4o', base_url=None, api_key=None, timeout_s=10, num_retries=0))
    with pytest.raises(StructuredLlmError):
        await client.evaluate_revision_work('fixture', response_format=SupportAssessmentWireResponse, max_tokens=1024)
