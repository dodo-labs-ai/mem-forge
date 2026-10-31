"""Compact transport retains exact Support coverage and evidence authority."""

from tests.evidence_display_fixture import evidence_displays
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from memforge.llm.structured import SupportAssessmentWireResponse
from memforge.pipeline.support_wire import SupportWireAliases

SUPPORTED = {
    'work_id': 'WRK-0001', 'status': 'supported', 'primary_ref': 'PRM-0002',
    'required_refs': ['REQ-0003'],
 "evidence_displays": evidence_displays('PRM-0002', ['REQ-0003'])}
CONTINUE = {
    'work_id': 'WRK-0001', 'status': 'continue',
    'witness_delta': {'support_witness_refs': ['PRM-0002'], 'opposing_witness_refs': ['REQ-0003']},
}


def wire():
    return SupportWireAliases(SimpleNamespace(fragments=[
        SimpleNamespace(reference='p000002', primary_eligible=True),
        SimpleNamespace(reference='r000003', primary_eligible=False),
    ]), [], {'w000001': 'WRK-0001'})


def row_schema():
    schema = SupportAssessmentWireResponse.model_json_schema()
    assert schema['properties']['results']['items'] == {'$ref': '#/$defs/SupportAssessmentWireRow'}
    return schema['$defs']['SupportAssessmentWireRow']


def test_every_status_answers_in_one_flat_row():
    row = row_schema()
    assert set(row['properties']) == {
        'work_id', 'status', 'witness_delta', 'primary_ref', 'required_refs', 'evidence_displays',
    }
    assert row['properties']['status']['enum'] == ['continue', 'supported', 'unsupported']
    assert row['additionalProperties'] is False
    with pytest.raises(ValidationError):
        SupportAssessmentWireResponse.model_validate({'results': [{**SUPPORTED, 'reason': 'A long explanation'}]})


def test_supported_row_decodes_to_canonical_refs():
    row = wire().decode(SupportAssessmentWireResponse.model_validate({'results': [SUPPORTED]})).results[0]
    assert row.work_id == 'w000001' and row.primary_ref == 'p000002'
    assert row.required_refs == ['r000003']


def test_continue_row_decodes_its_witness_delta():
    row = wire().decode(SupportAssessmentWireResponse.model_validate({'results': [CONTINUE]})).results[0]
    assert row.witness_delta.support_witness_refs == ['p000002']
    assert row.witness_delta.opposing_witness_refs == ['r000003']


@pytest.mark.parametrize('row, field', [
    ({**SUPPORTED, 'evidence_displays': None}, 'evidence_displays'),
    ({**SUPPORTED, 'witness_delta': CONTINUE['witness_delta']}, 'witness_delta'),
    ({**SUPPORTED, 'evidence_displays': evidence_displays('PRM-0002')}, 'evidence_displays'),
    ({**CONTINUE, 'witness_delta': None}, 'witness_delta'),
    ({**CONTINUE, 'primary_ref': 'PRM-0002'}, 'primary_ref'),
    ({'work_id': 'WRK-0001', 'status': 'unsupported', 'required_refs': []}, 'required_refs'),
])
def test_status_rules_reject_only_the_row_that_breaks_them(row, field):
    aliases = SupportWireAliases(SimpleNamespace(fragments=[
        SimpleNamespace(reference='p000002', primary_eligible=True),
        SimpleNamespace(reference='r000003', primary_eligible=False),
    ]), [], {'w000001': 'WRK-0001', 'w000002': 'WRK-0002'})
    response = SupportAssessmentWireResponse.model_validate({'results': [
        row, {'work_id': 'WRK-0002', 'status': 'unsupported'},
    ]})
    (bad, error), (good, decoded) = aliases.decode_rows(response)
    assert (bad, error.code.value, error.location) == ('w000001', 'invalid_selection', 'results[0]')
    assert field in str(error)
    assert good == 'w000002' and decoded.status == 'unsupported'


def test_strict_transport_requires_every_row_field():
    from litellm.utils import type_to_response_format_param
    from memforge.llm.structured import _native_schema_request_kwargs
    wire_format = type_to_response_format_param(SupportAssessmentWireResponse)['json_schema']
    row = wire_format['schema']['$defs']['SupportAssessmentWireRow']
    assert wire_format['strict'] is True and set(row['required']) == set(row['properties'])
    assert 'oneOf' not in json.dumps(wire_format)
    # An explicit wire contract sends the model's own schema: a closed row whose unused fields may be null.
    explicit = _native_schema_request_kwargs(SupportAssessmentWireResponse, 'json_schema_response_format')
    row = explicit['response_format']['json_schema']['schema']['$defs']['SupportAssessmentWireRow']
    assert row['additionalProperties'] is False and set(row['required']) == {'work_id', 'status'}
    assert {'type': 'null'} in row['properties']['primary_ref']['anyOf']


@pytest.mark.asyncio
async def test_live_client_contract_uses_compact_schema_and_rejects_truncated_json(monkeypatch):
    from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig, StructuredLlmError
    from tests.test_structured_llm import CompletionResponse, set_native_schema_support
    reply = CompletionResponse(json.dumps({'results': [{**SUPPORTED, 'required_refs': [],
        'evidence_displays': evidence_displays(SUPPORTED['primary_ref'])}]}))
    seen = []
    async def completion(**kwargs):
        seen.append(kwargs)
        return reply
    monkeypatch.setattr('memforge.llm.structured.litellm.acompletion', completion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(StructuredLlmConfig(
        model='openai/gpt-4o', base_url=None, api_key=None, timeout_s=10, num_retries=0))
    result = await client.evaluate_revision_work('fixture', response_format=SupportAssessmentWireResponse)
    assert len(result.results) == 1 and 'reason' not in result.results[0].model_dump()
    assert len(seen) == 1
    reply.choices[0].finish_reason = 'length'
    with pytest.raises(StructuredLlmError):
        await client.evaluate_revision_work('fixture', response_format=SupportAssessmentWireResponse)


@pytest.mark.asyncio
async def test_correction_log_names_the_rule_class_without_source_content(caplog):
    from memforge.pipeline.revision_work import RevisionWorkExecutor
    from tests.test_revision_work import CHANGED, Client, Store, work_items
    class WrongRef(Client):
        def judge(self, prompt):
            results = super().judge(prompt)
            results[0]['primary_ref'] = 'PRM-9999'
            results[0]['evidence_displays'] = evidence_displays('PRM-9999', results[0]['required_refs'])
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


def _object_unions(schema):
    """Paths of every anyOf or oneOf that offers more than one object shape."""
    defs = schema.get('$defs', {})

    def is_object(branch):
        branch = defs[branch['$ref'].rsplit('/', 1)[-1]] if '$ref' in branch else branch
        return 'properties' in branch

    def walk(node, path):
        if isinstance(node, dict):
            for key in ('anyOf', 'oneOf'):
                if sum(is_object(branch) for branch in node.get(key, [])) > 1:
                    yield path
            for key, value in node.items():
                yield from walk(value, f'{path}/{key}')
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from walk(value, f'{path}[{index}]')

    return list(walk(schema, ''))


def test_no_response_schema_asks_a_provider_to_choose_between_object_shapes():
    # Strict decoders on some OpenAI-compatible endpoints rarely reach every branch of
    # an object union, so a response row is one flat object and its decoder checks meaning.
    import inspect
    from pydantic import BaseModel
    from memforge.llm import capability, structured
    responses = [
        model for module in (structured, capability) for name, model in inspect.getmembers(module, inspect.isclass)
        if issubclass(model, BaseModel) and model.__module__ == module.__name__ and name.endswith('Response')
    ]
    assert SupportAssessmentWireResponse in responses
    assert {model.__name__: _object_unions(model.model_json_schema()) for model in responses} == {
        model.__name__: [] for model in responses
    }
