"""Compact transport retains exact Support coverage and evidence authority."""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from memforge.llm.structured import SupportAssessmentWireResponse
from memforge.pipeline.support_wire import SupportWireAliases


def test_success_schema_has_no_reason_and_decodes_to_canonical_result():
    result = SupportAssessmentWireResponse.model_validate({'results': [{
        'work_id': 'WRK-0001', 'status': 'supported', 'primary_ref': 'PRM-0002',
        'required_refs': ['REQ-0003'],
    }]})
    assert 'reason' not in result.results[0].model_dump()
    wire = SupportWireAliases(SimpleNamespace(fragments=[
        SimpleNamespace(reference='p000002', primary_eligible=True),
        SimpleNamespace(reference='r000003', primary_eligible=False),
    ]), [], {'w000001': 'WRK-0001'})
    row = wire.decode(result).results[0]
    assert row.work_id == 'w000001' and row.primary_ref == 'p000002'
    assert row.required_refs == ['r000003']
    assert row.reason == 'Supported by selected current Evidence.'
    schema = SupportAssessmentWireResponse.model_json_schema()
    success = next(v for v in schema['$defs'].values() if v.get('properties', {}).get('status', {}).get('const') == 'supported')
    assert 'reason' not in success['properties']
    assert success['additionalProperties'] is False


def test_negative_judgment_preserves_bounded_explanation():
    for status in ('insufficient', 'unsupported'):
        row = {'work_id': 'WRK-0001', 'status': status, 'primary_ref': None,
               'required_refs': [], 'reason': 'Current text omits the required scope.'}
        assert SupportAssessmentWireResponse.model_validate({'results': [row]}).results[0].reason == row['reason']
        row['reason'] = 'x' * 1001
        with pytest.raises(ValidationError):
            SupportAssessmentWireResponse.model_validate({'results': [row]})


def test_success_output_cannot_emit_prose():
    row = {'work_id': 'WRK-0001', 'status': 'supported', 'primary_ref': 'PRM-0002',
           'required_refs': [], 'reason': 'A long explanation'}
    with pytest.raises(ValidationError):
        SupportAssessmentWireResponse.model_validate({'results': [row]})

@pytest.mark.asyncio
async def test_live_client_contract_uses_compact_schema_and_rejects_truncated_json(monkeypatch):
    from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig, StructuredLlmError
    from tests.test_structured_llm import CompletionResponse, set_native_schema_support
    reply = CompletionResponse(json.dumps({'results': [{
        'work_id': 'WRK-0001', 'status': 'supported', 'primary_ref': 'PRM-0002', 'required_refs': [],
    }]}))
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
    from tests.test_revision_work import Client, Store, work_items
    class WrongRef(Client):
        def judge(self, prompt):
            results = super().judge(prompt)
            results[0].primary_ref = 'PRM-9999'
            return results
    executor = RevisionWorkExecutor(client=WrongRef(limit=50000), model='gpt-4o', store=Store(), derivation_id='root')
    with pytest.raises(Exception, match='bounded assessment correction exhausted'):
        await executor.assess_many(work_items('Two reviewers approve US releases.'))
    records = [r.message for r in caplog.records if r.message.startswith('llm_batch_output_rejected')]
    assert len(records) == 2
    assert all('FragmentSelectionError' in r and 'items=1' in r for r in records)
    assert all('reviewers' not in r and 'PRM-9999' not in r for r in records)


@pytest.mark.asyncio
async def test_pipeline_reuses_the_compact_wire_result_and_decodes_it_again():
    from memforge.pipeline.revision_work import RevisionWorkExecutor
    from tests.test_revision_work import Client, Store, work_items
    client, store = Client(limit=50000), Store()
    items = work_items('Two reviewers approve US releases.', 2)
    first = RevisionWorkExecutor(client=client, model='openai/gpt-4o', store=store, derivation_id='root')
    assert all(r.supported for r in (await first.assess_many(items)).values())
    second = RevisionWorkExecutor(client=client, model='openai/gpt-4o', store=store, derivation_id='root')
    reused = await second.assess_many(items)
    assert all(r.supported and r.reason == 'Supported by selected current Evidence.' for r in reused.values())
    assert len(client.prompts) == 1 and second.reused == 1
    work = next(w for w in store.works.values() if w.kind == 'support_assess')
    assert all('reason' not in r and r['work_id'].startswith('WRK-') for r in work.result['results'])
