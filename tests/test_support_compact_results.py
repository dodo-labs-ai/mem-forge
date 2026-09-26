"""Compact transport retains exact Support coverage and evidence authority."""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from memforge.llm.structured import SupportAssessmentWireResponse
from memforge.pipeline.support_wire import SupportWireAliases

SUPPORTED = {
    'work_id': 'WRK-0001', 'status': 'supported', 'primary_ref': 'PRM-0002',
    'required_refs': ['REQ-0003'],
}
CONTINUE = {
    'work_id': 'WRK-0001', 'status': 'continue',
    'witness_delta': {'support_witness_refs': ['PRM-0002'], 'opposing_witness_refs': ['REQ-0003']},
}


def wire():
    return SupportWireAliases(SimpleNamespace(fragments=[
        SimpleNamespace(reference='p000002', primary_eligible=True),
        SimpleNamespace(reference='r000003', primary_eligible=False),
    ]), [], {'w000001': 'WRK-0001'})


def row_schema(status):
    schema = SupportAssessmentWireResponse.model_json_schema()
    return next(v for v in schema['$defs'].values() if v.get('properties', {}).get('status', {}).get('const') == status)


def test_supported_row_has_no_prose_and_decodes_to_canonical_refs():
    result = SupportAssessmentWireResponse.model_validate({'results': [SUPPORTED]})
    row = wire().decode(result).results[0]
    assert row.work_id == 'w000001' and row.primary_ref == 'p000002'
    assert row.required_refs == ['r000003']
    success = row_schema('supported')
    assert set(success['properties']) == {'work_id', 'status', 'primary_ref', 'required_refs'}
    assert success['additionalProperties'] is False
    with pytest.raises(ValidationError):
        SupportAssessmentWireResponse.model_validate({'results': [{**SUPPORTED, 'reason': 'A long explanation'}]})


def test_unsupported_row_carries_no_refs_or_prose():
    assert set(row_schema('unsupported')['properties']) == {'work_id', 'status'}
    for extra in ({'primary_ref': 'PRM-0002'}, {'reason': 'Current text omits the required scope.'}):
        with pytest.raises(ValidationError):
            SupportAssessmentWireResponse.model_validate(
                {'results': [{'work_id': 'WRK-0001', 'status': 'unsupported', **extra}]}
            )


def test_continue_row_carries_only_its_witness_delta():
    assert set(row_schema('continue')['properties']) == {'work_id', 'status', 'witness_delta'}
    row = wire().decode(SupportAssessmentWireResponse.model_validate({'results': [CONTINUE]})).results[0]
    assert row.witness_delta.support_witness_refs == ['p000002']
    assert row.witness_delta.opposing_witness_refs == ['r000003']
    with pytest.raises(ValidationError):
        SupportAssessmentWireResponse.model_validate({'results': [{**CONTINUE, 'primary_ref': 'PRM-0002'}]})


def test_every_schema_property_is_required_for_strict_transport():
    for status in ('continue', 'supported', 'unsupported'):
        schema = row_schema(status)
        assert set(schema['required']) == set(schema['properties'])
    assert 'oneOf' not in json.dumps(SupportAssessmentWireResponse.model_json_schema())


@pytest.mark.asyncio
async def test_live_client_contract_uses_compact_schema_and_rejects_truncated_json(monkeypatch):
    from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig, StructuredLlmError
    from tests.test_structured_llm import CompletionResponse, set_native_schema_support
    reply = CompletionResponse(json.dumps({'results': [{**SUPPORTED, 'required_refs': []}]}))
    seen = []
    async def completion(**kwargs):
        seen.append(kwargs)
        return reply
    monkeypatch.setattr('memforge.llm.structured.litellm.acompletion', completion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(StructuredLlmConfig(
        model='openai/gpt-4o', base_url=None, api_key=None, timeout_s=10, num_retries=0))
    result = await client.evaluate_revision_work('fixture', response_format=SupportAssessmentWireResponse, max_tokens=1024)
    assert len(result.results) == 1 and 'reason' not in result.results[0].model_dump()
    assert len(seen) == 1
    reply.choices[0].finish_reason = 'length'
    with pytest.raises(StructuredLlmError):
        await client.evaluate_revision_work('fixture', response_format=SupportAssessmentWireResponse, max_tokens=1024)


@pytest.mark.asyncio
async def test_correction_log_names_the_rule_class_without_source_content(caplog):
    from memforge.pipeline.revision_work import RevisionWorkExecutor
    from tests.test_revision_work import CHANGED, Client, Store, work_items
    class WrongRef(Client):
        def judge(self, prompt):
            results = super().judge(prompt)
            results[0]['primary_ref'] = 'PRM-9999'
            return results
    executor = RevisionWorkExecutor(client=WrongRef(limit=50000), model='gpt-4o', store=Store(), derivation_id='root')
    [result] = (await executor.assess_many(work_items(CHANGED))).values()
    assert result.unresolved == 'invalid_response'
    records = [r.message for r in caplog.records if r.message.startswith('llm_batch_rows_rejected')]
    assert len(records) == 1
    assert all('FragmentSelectionError' in r and 'rejected=1' in r for r in records)
    assert all('reviewers' not in r and 'PRM-9999' not in r for r in records)


@pytest.mark.asyncio
async def test_pipeline_reuses_the_compact_wire_result_and_decodes_it_again():
    from memforge.pipeline.revision_work import RevisionWorkExecutor
    from tests.test_revision_work import CHANGED, Client, Store, work_items
    client, store = Client(limit=50000), Store()
    items = work_items(CHANGED, 2)
    first = RevisionWorkExecutor(client=client, model='openai/gpt-4o', store=store, derivation_id='root')
    assert all(r.supported for r in (await first.assess_many(items)).values())
    second = RevisionWorkExecutor(client=client, model='openai/gpt-4o', store=store, derivation_id='root')
    reused = await second.assess_many(items)
    assert all(r.supported and r.memory is not None for r in reused.values())
    # Both the Change Impact request and the reading request are reused without a call.
    assert len(client.impact_prompts) == 1 and len(client.prompts) == 1 and second.reused == 2
    work = next(w for w in store.works.values() if w.kind == 'support_assess')
    assert all('reason' not in r and r['work_id'].startswith('WRK-') for r in work.result['results'])
