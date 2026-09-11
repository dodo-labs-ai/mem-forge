"""Claim work recovery preserves full coverage without lifecycle side effects."""
from dataclasses import replace
import json

import pytest

from memforge.llm.structured import ClaimRevisionDecision, ClaimRevisionResponse, MemoryRelationAssessment, StructuredLlmError
from memforge.pipeline.claim_revision import assess_claim_pairs
from memforge.pipeline.reconciler import SupportAuditEntry, reconcile_memories
from tests.test_claim_revision import candidate, memory, Client
from tests.test_derivation_work import prepare_database


class PairClient(Client):
    max_concurrent = 1

    def __init__(self, fail_at=None):
        super().__init__("unrelated")
        self.fail_at = fail_at

    async def assess_claim_revisions(self, prompt, **kwargs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise StructuredLlmError("fixture deadline", terminal_category="deadline_exceeded", error_code="logical_deadline_exceeded")
        groups = json.loads(prompt.split("<memory_pair_groups>")[1].split("</memory_pair_groups>")[0])
        return ClaimRevisionResponse(decisions=[ClaimRevisionDecision(pair_index=item["pair_index"], status="resolved",
            relation=MemoryRelationAssessment(classification="unrelated", direction="symmetric", same_subject_and_scope=False,
                incompatible_assertions="")) for group in groups for item in group["candidates"]])


@pytest.mark.asyncio
async def test_last_of_23_claim_requests_resumes_after_database_reopen(tmp_path):
    from memforge.storage.database import Database
    path = tmp_path / "claim.db"
    db, root = await prepare_database(path)
    olds = [replace(memory(), id=f"old-{i:03}") for i in range(183)]
    kwargs = dict(candidates=[replace(candidate(), content=f"claim {i}") for i in range(8)], incumbents=olds,
        support_audits=[SupportAuditEntry(old.id, True) for old in olds], model="fixture",
        derivation_id=root.id, operation_input_hash="a" * 64)
    first = PairClient(fail_at=23)
    try:
        with pytest.raises(StructuredLlmError, match="fixture deadline"):
            await assess_claim_pairs(**kwargs, client=first, store=db)
        assert first.calls == 23
    finally:
        await db.close()
    db = Database(str(path))
    await db.connect()
    try:
        retry = PairClient()
        result = await assess_claim_pairs(**kwargs, client=retry, store=db)
        assert retry.calls == 1
        assert len(result.decisions) == 1464
        assert len({(c, m) for c, m, _ in result.decisions}) == 1464
        assert len(result.work_ids) == 23
        cursor = await db.db.execute("SELECT id FROM lifecycle_plans")
        assert not await cursor.fetchall()
        changed = PairClient()
        await assess_claim_pairs(**{**kwargs, "operation_input_hash": "b" * 64}, client=changed, store=db)
        assert changed.calls == 23
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_reconciliation_preserves_validation_fields_and_attempted_pair_count():
    class Invalid(Client):
        async def assess_claim_revisions(self, *args, **kwargs):
            raise StructuredLlmError("invalid", error_code="ValidationError",
                validation_fields=(("decisions.0.relation", "literal_error"),))
    result = await reconcile_memories(new_extractions=[candidate()], existing_memories=[memory()], doc_type="document",
        structured_llm_client=Invalid("unrelated"), support_audits=[SupportAuditEntry("memory", True)], include_metadata=True)
    assert not result.operations
    assert result.failure.validation_fields == (("decisions.0.relation", "literal_error"),)
    assert result.metrics.relation_pair_count == 1


@pytest.mark.parametrize("relation", ["equivalent", "unrelated", "refines_challenger_to_candidate", "refines_candidate_to_challenger", "contradicts", "insufficient"])
def test_compact_wire_preserves_relations_and_proofs(relation):
    from memforge.llm.structured import ClaimRevisionWireDecision, ClaimContradiction
    wire = ClaimRevisionWireDecision(pair_index=7, relation=relation,
        contradiction=ClaimContradiction(same_subject_and_scope=True, incompatible_assertions="one versus two") if relation == "contradicts" else None)
    decision = wire.decision()
    assert decision.pair_index == 7
    if relation == "insufficient":
        assert decision.status == "insufficient" and decision.relation is None
    elif relation.startswith("refines_"):
        assert decision.relation.classification == "refines"
        assert decision.relation.direction == relation.removeprefix("refines_")
    else:
        assert decision.relation.classification == relation
        assert decision.relation.direction == "symmetric"


def test_compact_unrelated_output_reduces_representation_without_dropping_slots():
    from memforge.llm.structured import ClaimRevisionWireResponse, ClaimRevisionWireDecision
    wire = ClaimRevisionWireResponse(decisions=[ClaimRevisionWireDecision(pair_index=i, relation="unrelated") for i in range(64)])
    expanded = ClaimRevisionResponse(decisions=[item.decision() for item in wire.decisions])
    assert len(wire.model_dump_json()) < len(expanded.model_dump_json()) * .65
    assert [x.pair_index for x in expanded.decisions] == list(range(64))


@pytest.mark.asyncio
async def test_provider_validation_diagnostic_reaches_durable_lifecycle_event(monkeypatch, tmp_path):
    from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig
    from memforge.evals.agent_evaluation import bind_source_lifecycle_outcome
    from tests.test_structured_llm import CompletionResponse
    async def invalid(**kwargs):
        return CompletionResponse('{"decisions":[{"pair_index":0,"relation":"invalid"}]}')
    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", invalid)
    client = LiteLlmStructuredClient(StructuredLlmConfig(model="openai/gpt-4o-mini", api_key="fixture",
        base_url=None, timeout_s=5, num_retries=0, native_schema_transport="response_format"))
    with pytest.raises(StructuredLlmError) as caught:
        await client.assess_claim_revisions("fixture", max_tokens=1024)
    error = caught.value
    assert error.validation_fields
    assert error.diagnostic.validation_location == "decisions.0.relation"
    bundle = bind_source_lifecycle_outcome(source_id="source-1", source_type="github_repo", doc_id="doc", source_unit_id="unit",
        base_unit_revision_id="base", target_unit_revision_id="target", projection_run_id="projection", operation_input_hash="a"*64,
        execution_owner_id="fixture-execution", outcome="failed", reason_code="relation_first_failed", attempt_count=1,
        duration_ms=10, incumbent_count=1, relation_pair_count=1, mutation_count=0, review_count=0, model_call_count=1,
        operation="assess_claim_revisions", terminal_category=error.terminal_category, error_code=error.error_code,
        validation_fields=error.validation_fields, diagnostic=error.diagnostic)
    assert bundle.event.validation_location == "decisions.0.relation"
    assert bundle.event.validation_rule == "literal_error"
    assert bundle.event.model == "openai/gpt-4o-mini"
    assert bundle.event.requested_max_tokens == 1024
    assert "invalid\"" not in str(bundle.event)
    from memforge.evals.agent_evaluation import AgentRuntimeEventQuery
    from datetime import timedelta
    db, _root = await prepare_database(tmp_path / "diagnostic.db")
    try:
        await db.record_agent_runtime_events(bundle.events)
        events = await db.list_agent_runtime_events(AgentRuntimeEventQuery(source_id="source-1",
            occurred_from=bundle.event.occurred_at - timedelta(seconds=1), occurred_to=bundle.event.occurred_at + timedelta(seconds=1)))
        assert events == list(bundle.events)
    finally:
        await db.close()
