"""Actual client telemetry survives a failed mandatory reconciliation stage."""

from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memforge.llm.structured import (
    LiteLlmStructuredClient,
    StructuredLlmConfig,
    StructuredLlmMetricsCollector,
)
from memforge.models import RawMemory
from memforge.pipeline.reconciler import reconcile_memories
from tests.test_relation_first_reconciliation import _memory, _relations_from_prompt
from tests.test_projected_lifecycle_integration import (
    db as db,
    _projection,
    _seed_incumbent_support,
    _candidate_retriever,
    _OutboxDrainer,
)


@pytest.mark.asyncio
async def test_mandatory_provider_failure_preserves_support_and_revision_without_plan(db, monkeypatch):
    from memforge.memory.engine import MemoryEngine, SourceUnitLifecycleExecutionError
    from memforge.storage.adapters.sqlite import build_sqlite_adapters

    async def unavailable(**kwargs):
        raise TimeoutError("private provider response")

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", unavailable)
    monkeypatch.setattr("memforge.llm.structured.litellm.supports_response_schema", lambda **_: False)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic/test",
            base_url=None,
            api_key=None,
            timeout_s=1,
            num_retries=0,
        )
    )
    first = _projection(run_id="diagnostics-before", body="A7 is removed.")
    await db.record_source_projection(first)
    incumbent = await _seed_incumbent_support(db, projection=first)
    await db.enable_lifecycle_gate("src-1")
    support_before = await db.get_active_memory_support_reference_ids(incumbent.id)
    second = _projection(
        run_id="diagnostics-after",
        body="A7 is retained.",
        prior=first.source_unit_revisions[0],
        prior_observations={first.observations[0].id: first.observation_revisions[0]},
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        cross_document_candidates=_candidate_retriever(adapters),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=client,
    )
    with pytest.raises(SourceUnitLifecycleExecutionError) as caught:
        await engine.prepare_and_commit_projected_lifecycle(
            projection=second,
            doc_id="confluence-123",
            raw_memories=[],
            doc_type="design-doc",
            project_key="ENG",
            repo_identifier=None,
            document_content=second.observation_revisions[0].content,
            update_mode="full_document",
            changed_hunks=None,
            update_plan_stats=None,
            source_updated_at=datetime.now(timezone.utc),
            lifecycle_execution_owner_id="diagnostics:attempt:1",
        )
    event = caught.value.runtime_bundle.event
    assert event.reason_code == "relation_first_failed"
    assert event.operation == "audit_incumbent_support"
    assert event.model_call_count == 1
    assert event.error_code == "TimeoutError"
    assert event.terminal_category == "provider_error"
    assert "private provider response" not in str(caught.value)
    assert await db.get_active_memory_support_reference_ids(incumbent.id) == support_before
    current = await db.get_current_source_unit_revision(first.source_units[0].id)
    assert current.id == first.source_unit_revisions[0].id
    assert (await db.db.execute_fetchall("SELECT COUNT(*) FROM lifecycle_plans"))[0][0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["classification", "support_audit"])
async def test_failed_second_call_is_counted_without_losing_unit_totals(monkeypatch, stage):
    calls = []

    async def provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise TimeoutError("provider secret detail must not be persisted")
        prompt = kwargs["messages"][0]["content"]
        if stage == "classification":
            payload = _relations_from_prompt(prompt).model_dump_json()
        else:
            incumbents = json.loads(prompt.split("<incumbents>")[1].split("</incumbents>")[0])
            payload = json.dumps({"decisions": [{"supported": True} for _ in incumbents]})
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=payload),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", provider)
    monkeypatch.setattr("memforge.llm.structured.litellm.supports_response_schema", lambda **_: False)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic/test",
            base_url=None,
            api_key=None,
            timeout_s=1,
            num_retries=0,
            max_concurrent=1,
        )
    )
    unit = StructuredLlmMetricsCollector()
    with client.metrics_scope(unit):
        result = await reconcile_memories(
            new_extractions=[RawMemory(content="New claim", memory_type="fact")] if stage == "classification" else [],
            existing_memories=[_memory(f"mem-{i}", f"Claim {i}") for i in range(65)],
            doc_type="design",
            structured_llm_client=client,
            include_metadata=True,
        )
    assert len(calls) == 2
    assert result.operations == []
    assert result.failure.reason_code == "relation_first_failed"
    assert result.metrics.structured_llm_calls == 2
    assert unit.summary(source_unit_elapsed_ms=1).logical_calls == 2
    assert result.failure.terminal_category == "provider_error"
    assert result.failure.error_code == "TimeoutError"
    assert result.failure.operation in {"classify_memory_relations", "audit_incumbent_support"}
    assert "provider secret" not in result.failure.error


def test_lifecycle_binder_retains_optional_safe_diagnostics():
    from memforge.evals.agent_evaluation import bind_source_lifecycle_outcome

    values = dict(
        source_id="src-1",
        source_type="jira",
        doc_id="jira-1",
        source_unit_id="unit-1",
        base_unit_revision_id=None,
        target_unit_revision_id="rev-1",
        projection_run_id="run-1",
        operation_input_hash="a" * 64,
        execution_owner_id="execution-1",
        outcome="failed",
        reason_code="relation_first_failed",
        attempt_count=1,
        duration_ms=10,
        incumbent_count=1,
        relation_pair_count=0,
        mutation_count=0,
        review_count=0,
        model_call_count=2,
    )
    legacy = bind_source_lifecycle_outcome(**values).event
    absent = bind_source_lifecycle_outcome(**values, operation=None, error_code=None, terminal_category=None).event
    assert legacy.payload_hash == absent.payload_hash
    diagnosed = bind_source_lifecycle_outcome(
        **values,
        operation="audit_incumbent_support",
        error_code="TimeoutError",
        terminal_category="provider_error",
    ).event
    assert diagnosed.event_id == legacy.event_id
    assert diagnosed.payload_hash != legacy.payload_hash
    assert diagnosed.operation == "audit_incumbent_support"
    assert diagnosed.error_code == "TimeoutError"
    assert diagnosed.terminal_category == "provider_error"


@pytest.mark.asyncio
async def test_failed_parallel_batch_counts_cancelled_provider_sibling(monkeypatch):
    started = 0
    both_started = asyncio.Event()

    async def provider(**kwargs):
        nonlocal started
        started += 1
        if started == 1:
            await both_started.wait()
            raise TimeoutError("first batch failed")
        both_started.set()
        await asyncio.Future()

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", provider)
    monkeypatch.setattr("memforge.llm.structured.litellm.supports_response_schema", lambda **_: False)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic/test",
            base_url=None,
            api_key=None,
            timeout_s=5,
            num_retries=0,
            max_concurrent=2,
        )
    )
    unit = StructuredLlmMetricsCollector()
    with client.metrics_scope(unit):
        result = await reconcile_memories(
            new_extractions=[RawMemory(content="New claim", memory_type="fact")],
            existing_memories=[_memory(f"mem-{i}", f"Claim {i}") for i in range(65)],
            doc_type="design",
            structured_llm_client=client,
            include_metadata=True,
        )
    summary = unit.summary(source_unit_elapsed_ms=1)
    assert started == 2
    assert result.operations == []
    assert result.failure.reason_code == "relation_first_failed"
    assert result.metrics.structured_llm_calls == 2
    assert summary.logical_calls == summary.provider_attempts == 2
    assert summary.terminal_category_counts == {"provider_error": 1, "cancelled": 1}
