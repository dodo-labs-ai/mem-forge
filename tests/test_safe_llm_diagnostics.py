import json

import pytest
from pydantic import BaseModel

from memforge.evals.agent_evaluation import QualitySignalCollector, quality_signal_scope
from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig, StructuredLlmError
from tests.test_structured_llm import CompletionResponse, set_native_schema_support


class ScoresResponse(BaseModel):
    scores: dict[str, int]


@pytest.mark.asyncio
@pytest.mark.parametrize('key', ['field with spaces', 'quote"/中文', 'x' * 300, 'plain'])
@pytest.mark.parametrize('recover', [True, False])
async def test_diagnostics_preserve_recovery_and_original_failure(monkeypatch, key, recover):
    from memforge.evals.agent_evaluation import _require_safe_diagnostic_path
    calls = []
    async def completion(**kwargs):
        calls.append(kwargs)
        value = 1 if recover and len(calls) > 1 else 'invalid'
        return CompletionResponse(json.dumps({'scores': {key: value}}))
    monkeypatch.setattr('memforge.llm.structured.litellm.acompletion', completion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(StructuredLlmConfig(
        model='openai/gpt-4o', base_url=None, api_key=None, timeout_s=10, num_retries=0))
    collector = QualitySignalCollector()
    with quality_signal_scope(collector):
        if recover:
            result = await client.evaluate_revision_work('fixture', response_format=ScoresResponse, max_tokens=1024)
            assert result.scores[key] == 1
        else:
            with pytest.raises(StructuredLlmError) as caught:
                await client.evaluate_revision_work('fixture', response_format=ScoresResponse, max_tokens=1024)
            assert caught.value.error_code == 'ValidationError'
            _require_safe_diagnostic_path('validation_location', caught.value.diagnostic.validation_location)
    attempts = [s for s in collector.snapshot() if s.event_name == 'structured_llm_attempt_outcome']
    assert attempts
    for signal in attempts:
        _require_safe_diagnostic_path('validation_location', signal.validation_location)
        assert signal.validation_rule == 'int_parsing'
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('recover', [True, False])
async def test_broken_diagnostic_reporting_cannot_change_call_outcome(monkeypatch, caplog, recover):
    async def completion(**kwargs):
        return CompletionResponse(json.dumps({'scores': {'field': 1 if recover else 'invalid'}}))
    def broken(*args, **kwargs):
        raise ValueError('PRIVATE_SOURCE_TEXT')
    monkeypatch.setattr('memforge.llm.structured.litellm.acompletion', completion)
    monkeypatch.setattr('memforge.evals.agent_evaluation.record_quality_signal', broken)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(StructuredLlmConfig(
        model='openai/gpt-4o', base_url=None, api_key=None, timeout_s=10, num_retries=0))
    if recover:
        result = await client.evaluate_revision_work('fixture', response_format=ScoresResponse, max_tokens=1024)
        assert result.scores['field'] == 1
    else:
        with pytest.raises(StructuredLlmError) as caught:
            await client.evaluate_revision_work('fixture', response_format=ScoresResponse, max_tokens=1024)
        assert caught.value.error_code == 'ValidationError'
    assert 'PRIVATE_SOURCE_TEXT' not in caplog.text


def test_bounded_coordinate_preserves_normal_paths_and_redacts_unsafe_segments():
    from memforge.diagnostics import diagnostic_path, is_diagnostic_path
    assert diagnostic_path(('memories', 0, 'content')) == 'memories.0.content'
    assert diagnostic_path(()) == '$'
    assert diagnostic_path(('scores', 'private key/中文')) == 'scores._'
    assert diagnostic_path(('scores', 'x' * 300)) == 'scores._'
    long = diagnostic_path(['field'] * 100)
    assert is_diagnostic_path(long) and len(long) == 255 and long.endswith('...')
