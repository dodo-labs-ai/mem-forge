import asyncio
import json

import pytest
import litellm

from memforge.llm.request_budget import RequestBudget
from memforge.llm.structured import (
    ClaimRevisionWireDecision, LiteLlmStructuredClient, RerankResponse, StructuredLlmConfig,
)


class Sink:
    def __init__(self):
        self.records = {}

    def write(self, trace_id, payload):
        self.records[trace_id] = json.loads(payload)
        return f"test://{trace_id}"


def client(sink=None):
    return LiteLlmStructuredClient(
        StructuredLlmConfig(model="openai/test", base_url=None, api_key="secret", timeout_s=2,
                            native_schema_transport="json_schema_response_format"),
        failure_trace_sink=sink,
    )


def response(text):
    return litellm.ModelResponse(choices=[{"message": {"content": text}, "finish_reason": "stop"}])


@pytest.mark.asyncio
async def test_failed_attempt_retains_full_io_even_when_fallback_recovers(monkeypatch):
    from memforge.llm.failure_trace import failure_trace_context
    sink = Sink()
    replies = iter(["{broken", '{"results":[]}'])
    async def complete(**kwargs):
        return response(next(replies))
    monkeypatch.setattr("litellm.acompletion", complete)
    prompt = "完整输入" * 100000
    with failure_trace_context(doc_id="doc-1", trace_id="a" * 32):
        await client(sink)._call_schema(prompt=prompt, response_format=RerankResponse, max_tokens=100)
    record = next(iter(sink.records.values()))
    assert record["prompt"] == prompt
    assert record["trace_id"] == "a" * 32
    assert record["lineage"]["doc_id"] == "doc-1"
    assert record["attempts"][0]["request"]["messages"][0]["content"] == prompt
    assert record["attempts"][0]["response"]["choices"][0]["message"]["content"] == "{broken"
    assert record["outcome"] == "recovered"
    assert "secret" not in json.dumps(record)


@pytest.mark.asyncio
async def test_business_validation_failure_retains_original_provider_response(monkeypatch):
    from memforge.llm.failure_trace import validation_trace
    sink = Sink()
    raw = '{"results":[]}'
    async def complete(**kwargs):
        return response(raw)
    monkeypatch.setattr("litellm.acompletion", complete)
    parsed = await client(sink)._call_schema(prompt="request", response_format=RerankResponse, max_tokens=100)
    assert not sink.records
    with pytest.raises(ValueError, match="unknown ref"):
        async with validation_trace(parsed, work_id="work-1"):
            raise ValueError("unknown ref")
    record = next(iter(sink.records.values()))
    assert record["attempts"][0]["response"]["choices"][0]["message"]["content"] == raw
    assert record["failures"][-1]["stage"] == "business_validation"
    assert record["lineage"]["work_id"] == "work-1"


@pytest.mark.asyncio
async def test_trace_sink_failure_preserves_original_validation_error(monkeypatch):
    from memforge.llm.failure_trace import validation_trace
    class BrokenSink:
        def write(self, *args):
            raise RuntimeError("storage unavailable")
    async def complete(**kwargs):
        return response('{"results":[]}')
    monkeypatch.setattr("litellm.acompletion", complete)
    parsed = await client(BrokenSink())._call_schema(prompt="request", response_format=RerankResponse, max_tokens=100)
    with pytest.raises(ValueError, match="original"):
        async with validation_trace(parsed):
            raise ValueError("original")


@pytest.mark.asyncio
async def test_success_and_disabled_capture_create_no_artifacts(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMFORGE_LLM_FAILURE_CAPTURE_ENABLED", raising=False)
    monkeypatch.setenv("MEMFORGE_LLM_FAILURE_TRACE_DIR", str(tmp_path))
    async def complete(**kwargs):
        return response('{"results":[]}')
    monkeypatch.setattr("litellm.acompletion", complete)
    sink = Sink()
    await client(sink)._call_schema(prompt="request", response_format=RerankResponse, max_tokens=100)
    await client()._call_schema(prompt="request", response_format=RerankResponse, max_tokens=100)
    assert not sink.records
    assert not list(tmp_path.iterdir())


def test_schema_explains_inapplicable_proofs_and_still_rejects_them():
    props = ClaimRevisionWireDecision.model_json_schema()["properties"]
    assert "null" in props["revision_assessment"]["description"]
    assert "refines_challenger_to_candidate" in props["revision_assessment"]["description"]
    assert "null" in props["contradiction"]["description"]
    with pytest.raises(ValueError, match="revision proof"):
        ClaimRevisionWireDecision.model_validate({"existing_id":"MEM-0000", "relation":"equivalent",
            "revision_assessment": {"same_knowledge_item": True, "preserves_incumbent_truth": True,
                "challenger_is_complete_current_claim": True}})


@pytest.mark.parametrize("field", ["primary_ref", "required_refs"])
def test_unknown_support_reference_identifies_exact_field_and_allowed_catalog(field):
    from types import SimpleNamespace
    from memforge.pipeline.support_wire import SupportWireAliases
    from memforge.llm.structured import SupportAssessmentWireResponse
    aliases = SupportWireAliases(SimpleNamespace(fragments=[SimpleNamespace(reference="f2", primary_eligible=True)]), [], {"work":"WRK-0000"})
    row = dict(work_id="WRK-0000", status="supported", primary_ref="PRM-0002", required_refs=[], omitted_matched_refs=[])
    row[field] = "PRM-0007" if field == "primary_ref" else ["PRM-0007"]
    with pytest.raises(ValueError) as error:
        aliases.decode(SupportAssessmentWireResponse.model_validate({"results":[row]}))
    expected = "results[0].primary_ref" if field == "primary_ref" else "results[0].required_refs[0]"
    assert error.value.location == expected
    assert error.value.received == "PRM-0007"
    assert error.value.allowed_refs == ["PRM-0002"]


def test_unknown_primary_diagnostic_excludes_required_only_refs():
    from types import SimpleNamespace
    from memforge.pipeline.support_wire import SupportWireAliases
    from memforge.llm.structured import SupportAssessmentWireResponse
    catalog = SimpleNamespace(fragments=[SimpleNamespace(reference="f2", primary_eligible=True),
        SimpleNamespace(reference="f3", primary_eligible=False)])
    aliases = SupportWireAliases(catalog, [], {"work":"WRK-0000"})
    with pytest.raises(ValueError) as error:
        aliases.decode(SupportAssessmentWireResponse.model_validate({"results":[dict(
            work_id="WRK-0000", status="supported", primary_ref="PRM-0007", required_refs=[],
            omitted_matched_refs=[])]}))
    assert error.value.allowed_refs == ["PRM-0002"]


@pytest.mark.asyncio
async def test_capture_preserves_non_exception_base_exception(monkeypatch):
    async def complete(**kwargs):
        raise SystemExit("shutdown")
    monkeypatch.setattr("litellm.acompletion", complete)
    with pytest.raises(SystemExit, match="shutdown"):
        await client(Sink())._call_schema(prompt="request", response_format=RerankResponse, max_tokens=100)


@pytest.mark.asyncio
async def test_actual_logical_deadline_artifact_records_terminal_reason(monkeypatch):
    from dataclasses import replace
    sink = Sink()
    slow = client(sink)
    slow.config = replace(slow.config, timeout_s=0.05)
    async def complete(**kwargs):
        await asyncio.sleep(1)
    monkeypatch.setattr("litellm.acompletion", complete)
    with pytest.raises(Exception):
        await slow._call_schema(prompt="request", response_format=RerankResponse, max_tokens=100)
    record = next(iter(sink.records.values()))
    assert record["failures"][-1]["error_code"] == "logical_deadline_exceeded"
    assert record["attempts"][0]["request"]["timeout"] > 0
    assert record["attempts"][0]["request"]["num_retries"] == 0


@pytest.mark.asyncio
async def test_concurrent_validation_capture_keeps_each_documents_own_io(monkeypatch):
    from memforge.llm.failure_trace import failure_trace_context, validation_trace
    sink = Sink()
    async def complete(**kwargs):
        await asyncio.sleep(0)
        return response('{"ranking":[]}')
    monkeypatch.setattr("litellm.acompletion", complete)
    shared = client(sink)
    async def run(doc):
        with failure_trace_context(doc_id=doc):
            parsed = await shared._call_schema(prompt=doc, response_format=RerankResponse, max_tokens=100)
            with pytest.raises(ValueError):
                async with validation_trace(parsed):
                    raise ValueError(doc)
    await asyncio.gather(run("doc-a"), run("doc-b"))
    assert {r["lineage"]["doc_id"]:r["prompt"] for r in sink.records.values()} == {"doc-a":"doc-a", "doc-b":"doc-b"}


def test_failure_trace_uses_the_same_lifecycle_identity_as_online_events():
    from memforge.evals.agent_evaluation import bind_source_lifecycle_outcome, source_lifecycle_execution_identity
    args = dict(source_id="src-1", source_unit_id="unit-1", base_unit_revision_id="base-1",
        target_unit_revision_id="rev-1", operation_input_hash="a"*64, execution_owner_id="owner-1")
    identity = source_lifecycle_execution_identity(**args)
    event = bind_source_lifecycle_outcome(**args, source_type="jira", doc_id="doc-1",
        projection_run_id="run-1", outcome="failed", reason_code="reconciliation_failed",
        attempt_count=1, duration_ms=0, incumbent_count=0, relation_pair_count=0,
        mutation_count=0, review_count=0, model_call_count=1).event
    assert identity == dict(operation_id=event.operation_id, execution_id=event.execution_id, trace_id=event.trace_id)


@pytest.mark.asyncio
async def test_local_timeout_artifact_is_complete_and_marks_missing_output(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMFORGE_LLM_FAILURE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("MEMFORGE_LLM_FAILURE_TRACE_DIR", str(tmp_path))
    async def complete(**kwargs):
        raise TimeoutError("transport timeout")
    monkeypatch.setattr("litellm.acompletion", complete)
    with pytest.raises(Exception):
        await client()._call_schema(prompt="full request", response_format=RerankResponse, max_tokens=100,
            retry_with_json_text=False)
    record = json.loads(next(tmp_path.iterdir()).read_bytes())
    assert record["prompt"] == "full request"
    assert record["attempts"] and all(a["response"] is None for a in record["attempts"])


@pytest.mark.asyncio
async def test_optional_selector_correction_keeps_rejected_extraction_trace(monkeypatch):
    from memforge.pipeline.fragment_selector_correction import correct_fragment_selectors_once
    from memforge.pipeline.projection_fragments import FragmentSelectionError, FragmentSelectionErrorCode
    from memforge.llm.structured import ProjectionFragmentMemoryCandidate
    sink = Sink()
    async def complete(**kwargs):
        return response('{"ranking":[]}')
    monkeypatch.setattr("litellm.acompletion", complete)
    source_response = await client(sink)._call_schema(prompt="original extraction", response_format=RerankResponse, max_tokens=100)
    class Catalog:
        def resolve_selection(self, **kwargs):
            raise FragmentSelectionError(FragmentSelectionErrorCode.UNKNOWN_REF, "unknown selector")
    class CorrectionClient:
        def request_budget(self, model=None):
            return RequestBudget("fixture", 200000, 200000, 64000, 0.8, "fixture")

        def request_fits(self, *args, **kwargs):
            return False

        async def correct_projection_fragment_selectors(self, *args, **kwargs):
            raise AssertionError("a correction that does not fit is never sent")
    candidate = ProjectionFragmentMemoryCandidate(content="fixed claim", memory_type="fact", primary_ref="unknown")
    _, metrics = await correct_fragment_selectors_once([candidate], catalog=Catalog(), client=CorrectionClient(),
        extraction_prompt="original extraction", max_tokens=100, model=None, images=(), source_response=source_response)
    assert metrics["selector_correction_outcome"] == "capacity_skipped"
    record = next(iter(sink.records.values()))
    assert record["prompt"] == "original extraction"
    assert record["failures"][-1]["stage"] == "business_validation"


@pytest.mark.asyncio
async def test_actual_executor_correction_marks_failed_attempt_recovered(monkeypatch):
    from memforge.pipeline.revision_work import RevisionWorkExecutor
    from tests.test_revision_work import Client, work_items, payload
    sink = Sink()
    calls = 0
    actual = LiteLlmStructuredClient(StructuredLlmConfig(model="openai/gpt-4o", base_url=None,
        api_key="secret", timeout_s=2, native_schema_transport="json_schema_response_format"), failure_trace_sink=sink)
    async def complete(**kwargs):
        nonlocal calls
        calls += 1
        p = payload(kwargs["messages"][0]["content"])
        work_id = p["works"][0]["work_id"]
        row = (dict(work_id=work_id, status="unsupported") if p["last"] else dict(
            work_id=work_id, status="continue",
            witness_delta=dict(support_witness_refs=["PRM-9999"] if calls == 1 else [], opposing_witness_refs=[])))
        return response(json.dumps({"results":[row]}))
    monkeypatch.setattr("litellm.acompletion", complete)
    class ExecutorClient(Client):
        async def evaluate_revision_work(self, prompt, **kwargs):
            return await actual.evaluate_revision_work(prompt, **kwargs)
    executor = RevisionWorkExecutor(client=ExecutorClient(limit=100000), model="openai/gpt-4o")
    await executor.assess_many(work_items("Two reviewers approve US releases.\nRoutine note."))
    # The first request is corrected once; the reading then continues to its last group.
    assert calls == 3
    record = next(iter(sink.records.values()))
    assert record["outcome"] == "recovered"
    assert record["failures"][-1]["location"] == "results[0].support_witness_refs[0]"
    assert "PRM-9999" in record["attempts"][0]["response"]["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_multiple_selector_errors_preserve_original_row_and_field_indices(monkeypatch):
    from memforge.pipeline.fragment_selector_correction import correct_fragment_selectors_once
    from memforge.pipeline.projection_fragments import FragmentSelectionError, FragmentSelectionErrorCode
    from memforge.llm.structured import ProjectionFragmentMemoryCandidate
    sink = Sink()
    async def complete(**kwargs):
        return response('{"ranking":[]}')
    monkeypatch.setattr("litellm.acompletion", complete)
    original = await client(sink)._call_schema(prompt="original", response_format=RerankResponse, max_tokens=100)
    class Catalog:
        def resolve_selection(self, **kwargs):
            raise FragmentSelectionError(FragmentSelectionErrorCode.UNKNOWN_REF, "unknown selector",
                location="required_refs[0]", received=kwargs["required_refs"][0], allowed_refs=["p2"])
    class CorrectionClient:
        def request_budget(self, model=None):
            return RequestBudget("fixture", 200000, 200000, 64000, 0.8, "fixture")

        def request_fits(self, *args, **kwargs):
            return False

        async def correct_projection_fragment_selectors(self, *args, **kwargs):
            raise AssertionError("a correction that does not fit is never sent")
    candidates = [ProjectionFragmentMemoryCandidate(content=f"claim {i}", memory_type="fact", primary_ref="p2",
        required_refs=["p2", f"unknown-{i}", f"unknown-{i}"]) for i in range(2)]
    await correct_fragment_selectors_once(candidates, catalog=Catalog(), client=CorrectionClient(),
        extraction_prompt="original", max_tokens=100, model=None, images=(), source_response=original)
    errors = next(iter(sink.records.values()))["failures"]
    assert [e["location"] for e in errors] == ["memories[0].required_refs[1]", "memories[1].required_refs[1]"]
    assert [e["context"]["candidate_index"] for e in errors] == [0, 1]
