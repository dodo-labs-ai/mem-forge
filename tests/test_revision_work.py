"""Transport and lifecycle contracts; fixture judgments are not model-accuracy evidence."""

import json
from dataclasses import replace

import pytest

from memforge.pipeline.revision_work import RevisionWorkExecutor, SupportWorkItem
from memforge.llm.structured import SupportAssessmentResponse, SupportAssessmentResult
from tests.test_revision_assessment import revisions, old_support, memory


class Client:
    def __init__(self, limit=8000):
        self.limit = limit
        self.prompts = []
        self.fail_at = None

    def request_budget(self, model=None):
        from memforge.llm.request_budget import RequestBudget
        return RequestBudget(model or "fixture", self.limit, self.limit * 8, 32768, 1, "fixture")

    def request_fits(self, prompt, *, max_tokens, reserve_correction=True, **kwargs):
        return len(prompt) + max_tokens // 8 <= self.limit - (256 if reserve_correction else 0)

    def request_tokens(self, prompt, **kwargs):
        return len(prompt)

    def input_policy_identity_for(self, model=None):
        return f"fixture-{self.limit}-{model}"

    async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
        assert response_format is SupportAssessmentResponse
        self.prompts.append(prompt)
        if self.fail_at == len(self.prompts):
            self.fail_at = None
            raise TimeoutError("fixture interruption")
        payload = json.loads(prompt.split("<assessment>")[1].split("</assessment>")[0])
        rows = payload["current"]["primary_candidates"] + payload["current"]["required_only_candidates"]
        previous = {r["work_id"]: r for r in payload["previous_state"]}
        results = []
        for claim in payload["claims"]:
            state = SupportAssessmentResult.model_validate(previous[claim["work_id"]])
            facts = [state.reason]
            for ref, text, *_ in rows:
                if any(term in text for term in ("Cedar", "Alder", "Birch")):
                    if text not in facts:
                        facts.append(text)
                if "Two reviewers approve US releases." == text.strip():
                    state.primary_ref = ref
            combined = " ".join(facts)
            exception = ("Cedar uses one reviewer" in combined and "Cedar means US" in combined) or all(
                part in combined for part in ("Alder releases need not", "Alder refers to Birch", "Birch refers to US")
            )
            if exception:
                state.status = "unsupported"
            elif state.status != "unsupported":
                state.status = "supported" if state.primary_ref else "insufficient"
            state.reason = " ".join(facts)
            results.append(state)
        return SupportAssessmentResponse(results=results)


class Store:
    def __init__(self):
        self.works = {}

    async def stage_derivation_work(self, *, derivation_id, work):
        return self.works.setdefault((derivation_id, work.id), work)

    async def record_derivation_work(self, *, derivation_id, work):
        self.works[derivation_id, work.id] = work
        return work


def work_items(text, count=1):
    from memforge.pipeline.revision_assessment import RevisionAssessmentContext

    base, current = revisions("Two reviewers approve US releases.\n", text)
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    return [
        SupportWorkItem(f"w{i}", replace(memory(), id=f"memory-{i}"), old_support(base), context) for i in range(count)
    ]


def payload(prompt):
    return json.loads(prompt.split("<assessment>")[1].split("</assessment>")[0])


@pytest.mark.asyncio
async def test_small_delta_shares_one_direct_request_and_program_completion():
    client, store = Client(limit=16000), Store()
    items = work_items("Two reviewers approve US releases.\n", 3)
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    results = await executor.assess_many(items)
    assert len(client.prompts) == 1 and len(results) == 3
    assert all(r.supported for r in results.values())
    assert payload(client.prompts[0])["input_mode"] == "delta"
    assert set(payload(client.prompts[0])["previous_state"][0]) == {
        "work_id", "status", "reason", "primary_ref", "required_refs"
    }
    receipts = [w for w in store.works.values() if w.kind == "support_finalize"]
    assert len(receipts) == 1 and receipts[0].manifest["completion"] == "program"
    assert receipts[0].manifest["dependencies"] and executor.calls == 1
    assert all(
        r.memory.resolved_evidence_selection.parts[0].anchor.observation_revision_id == "rev-primary-v2"
        for r in results.values()
    )


@pytest.mark.asyncio
async def test_small_change_never_sends_unchanged_document_even_when_it_fits():
    from memforge.pipeline.revision_assessment import RevisionAssessmentContext

    body = "Two reviewers approve US releases.\n\n" + "\n\n".join(f"Unchanged unrelated note {i}." for i in range(100))
    base, current = revisions(body, body + "\n\nNew unrelated execution record.")
    item = SupportWorkItem(
        "w0",
        memory(),
        old_support(base),
        RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope"),
    )
    client = Client(limit=100000)
    await RevisionWorkExecutor(client=client, model="fixture").assess_many([item])
    assert len(client.prompts) == 1
    assert "New unrelated execution record" in client.prompts[0]
    assert "Unchanged unrelated note 99" not in client.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_cross_batch_exception_survives_later_unrelated_and_rule_text(reverse):
    pieces = ["Cedar uses one reviewer.", "Cedar means US releases.", "Two reviewers approve US releases."]
    if reverse:
        pieces.reverse()
    filler = "\n\n".join(f"Routine execution note {i}: maintain settings." for i in range(100))
    text = ("\n\n" + filler + "\n\n").join(pieces)
    client = Client()
    items = work_items(text, 3)
    executor = RevisionWorkExecutor(client=client, model="fixture")
    results = await executor.assess_many(items)
    assert all(not r.supported for r in results.values())
    assert len(client.prompts) > 2
    assert all("<scan>" not in p and "<reduce>" not in p and "<final>" not in p for p in client.prompts)
    for work_id in results:
        spans = [
            payload(p)["coverage"] for p in client.prompts if any(c["work_id"] == f"WRK-{int(work_id[1:]):04d}" for c in payload(p)["claims"])
        ]
        assert spans[0]["processed_before"] == 0
        assert spans[-1]["complete_after_batch"]
        assert all(a["processed_before"] + a["batch_size"] == b["processed_before"] for a, b in zip(spans, spans[1:]))
    assert any(len(payload(p)["claims"]) > 1 for p in client.prompts)


@pytest.mark.asyncio
async def test_retry_reuses_completed_assessments_without_replaying_their_model_calls():
    items = work_items("Two reviewers approve US releases.\n\n" + "\n\n".join(f"Routine note {i}." for i in range(350)))
    client, store = Client(), Store()
    client.fail_at = 2
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    with pytest.raises(TimeoutError):
        await executor.assess_many(items)
    done = {w.id: w.result_hash for w in store.works.values() if w.status == "completed"}
    assert done and not executor.final_work_ids
    retry = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    assert (await retry.assess_many(items))["w0"].supported
    assert retry.reused == len(done)
    assert all(store.works["root", key].result_hash == value for key, value in done.items())


@pytest.mark.asyncio
async def test_unknown_selector_gets_one_local_correction_not_a_completed_receipt():
    class InvalidClient(Client):
        async def evaluate_revision_work(self, prompt, **kwargs):
            result = await super().evaluate_revision_work(prompt, **kwargs)
            result.results[0].primary_ref = "not-supplied"
            return result

    client = InvalidClient()
    executor = RevisionWorkExecutor(client=client, model="fixture")
    with pytest.raises(Exception, match="bounded assessment correction exhausted"):
        await executor.assess_many(work_items("Two reviewers approve US releases."))
    assert len(client.prompts) == 2 and "Correction:" in client.prompts[-1]
    assert not executor.final_work_ids


@pytest.mark.asyncio
async def test_unknown_baseline_uses_full_batches_without_inheriting_historical_evidence():
    from memforge.pipeline.revision_assessment import RevisionAssessmentContext

    items = work_items("Two reviewers approve US releases.\n\n" + "\n\n".join(f"Routine note {i}." for i in range(300)))
    old = items[0]
    context = RevisionAssessmentContext(projection=old.context.projection, base=None, access_context_hash="scope")
    item = replace(
        old,
        context=context,
        support=tuple(replace(p, excerpt="huge historical material " * 10000) for p in old.support),
    )
    client = Client()
    result = await RevisionWorkExecutor(client=client, model="fixture").assess_many([item])
    assert result["w0"].supported and len(client.prompts) > 1
    assert all(payload(p)["input_mode"] == "full" and "historical_evidence" not in payload(p) for p in client.prompts)
    assert payload(client.prompts[0])["previous_state"][0]["status"] == "insufficient"


@pytest.mark.asyncio
async def test_harmless_long_reason_does_not_fail_schema_validation():
    row = SupportAssessmentResult(work_id="w0", status="insufficient", reason="Explanation " * 300)
    assert len(row.reason) > 1000


def test_previous_evidence_uses_current_ref_only_for_exact_current_anchor_and_text():
    items = work_items("Two reviewers approve US releases.\n")
    item = items[0]
    catalog = item.context.catalog(item.context.full_fragments)
    fragment = next(f for f in catalog.fragments if "Two reviewers" in f.presentation_text)
    current_part = replace(item.support[0], anchor=fragment.anchor, excerpt=fragment.presentation_text)
    payload = RevisionWorkExecutor._previous_evidence([replace(item, support=(current_part,))], catalog)
    assert payload["historical_evidence"] == []
    assert payload["previous_evidence"][0]["parts"] == [{"role": "primary", "current_ref": fragment.reference}]
    # A matching anchor with different text is not current proof.
    mismatched = replace(current_part, excerpt="Two reviewers approve EU releases.")
    payload = RevisionWorkExecutor._previous_evidence([replace(item, support=(mismatched,))], catalog)
    assert payload["previous_evidence"][0]["parts"] == [{"role": "primary", "historical_index": 0}]
    assert payload["historical_evidence"][0]["excerpt"] == mismatched.excerpt


@pytest.mark.asyncio
async def test_growing_shared_state_splits_only_unprocessed_tail_and_resumes():
    class GrowingClient(Client):
        def request_fits(self, prompt, **kwargs):
            data = payload(prompt)
            states = data["previous_state"]
            if len(states) > 1 and any(len(s["reason"]) > 1000 for s in states):
                return False
            if data["coverage"]["batch_size"] > 12:
                return False
            return super().request_fits(prompt, **kwargs)

        async def evaluate_revision_work(self, prompt, **kwargs):
            result = await super().evaluate_revision_work(prompt, **kwargs)
            for r in result.results:
                r.reason = "Important cumulative interpretation. " * 45
            return result

    items = work_items(
        "Two reviewers approve US releases.\n\n" + "\n\n".join(f"Routine note {i}." for i in range(70)), 2
    )
    store, client = Store(), GrowingClient(limit=20000)
    client.fail_at = 3
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    with pytest.raises(TimeoutError):
        await executor.assess_many(items)
    first = payload(client.prompts[0])
    assert len(first["claims"]) == 2
    prefix = first["coverage"]["batch_size"]
    retry_client = GrowingClient(limit=20000)
    retry = RevisionWorkExecutor(client=retry_client, model="fixture", store=store, derivation_id="root")
    assert all(r.supported for r in (await retry.assess_many(items)).values())
    assert retry.reused >= 2
    assert all(payload(p)["coverage"]["processed_before"] >= prefix for p in retry_client.prompts)
    assert all(len(payload(p)["claims"]) == 1 for p in client.prompts[1:] + retry_client.prompts)
    receipts = [w for w in store.works.values() if w.kind == "support_finalize"]
    assert len(receipts) == 2
    assert receipts[0].manifest["dependencies"][0] == receipts[1].manifest["dependencies"][0]


def test_output_allowance_saturates_provider_capacity_without_limiting_refs():
    items = work_items("Two reviewers approve US releases.", 32)
    executor = RevisionWorkExecutor(client=Client(), model="gpt-4o")
    assert executor._output(items, 300) == executor.client.request_budget().output_reserve(999999)


def test_deleted_text_counts_as_input_but_never_as_selectable_output_refs():
    from memforge.pipeline.revision_assessment import RevisionAssessmentContext

    previous = "Two reviewers approve US releases.\n\n" + "\n\n".join(f"Deleted condition {i}." for i in range(250))
    base, target = revisions(previous, "Two reviewers approve US releases.")
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash="scope")
    items = [SupportWorkItem(f"w{i}", replace(memory(), id=f"m{i}"), old_support(base), context) for i in range(3)]
    executor = RevisionWorkExecutor(client=Client(limit=100000), model="gpt-4o")
    scope = executor._range(items)
    states = {i.id: executor._initial(scope, i) for i in items}
    units = [("historical", part) for part in scope.removed]
    prompt, catalog, budget = executor._request(scope, units, items, states, 0, len(units))
    assert len(units) >= 250
    assert "Deleted condition 249" in prompt
    assert budget == executor._output(items, len(catalog.fragments), states.values())


@pytest.mark.asyncio
async def test_optional_history_cannot_block_current_full_structure():
    from memforge.pipeline.revision_assessment import RevisionAssessmentContext

    item = work_items("Two reviewers approve US releases.\n\n" + "A normal current paragraph. " * 110)[0]
    item = replace(
        item,
        support=tuple(replace(p, excerpt="historical background " * 190) for p in item.support),
        context=RevisionAssessmentContext(projection=item.context.projection, base=None, access_context_hash="scope"),
    )
    client = Client(limit=10000)
    result = await RevisionWorkExecutor(client=client, model="fixture").assess_many([item])
    assert result[item.id].supported
    assert any("historical_evidence" not in payload(p) for p in client.prompts)


@pytest.mark.asyncio
async def test_insufficient_is_preserved_in_complete_receipt_without_current_evidence():
    class InsufficientClient(Client):
        async def evaluate_revision_work(self, prompt, **kwargs):
            result = await super().evaluate_revision_work(prompt, **kwargs)
            result.results[0].status = "insufficient"
            result.results[0].reason = "Cannot establish the remaining condition"
            return result

    client, store = InsufficientClient(limit=16000), Store()
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    results = await executor.assess_many(work_items("Two reviewers approve US releases.\n", 2))
    assert results["w0"].supported is None
    assert results["w0"].memory is None
    assert results["w1"].supported is True
    assert len(client.prompts) == 1
    assert len(executor.final_work_ids) == 1
    receipt = store.works["root", executor.final_work_ids[0]]
    assert receipt.status == "completed"
    assert receipt.result["results"][0]["status"] == "insufficient"
    assert receipt.manifest["coverage"]["work_ids"] == ["w0", "w1"]


@pytest.mark.asyncio
async def test_claim_can_select_current_evidence_carried_by_another_claim_in_same_request():
    class SharedEvidenceClient(Client):
        def __init__(self):
            super().__init__()
            self.borrowed = False

        async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
            self.prompts.append(prompt)
            data = payload(prompt)
            rows = data["current"]["primary_candidates"]
            previous = {row["work_id"]: row for row in data["previous_state"]}
            states = {key: SupportAssessmentResult.model_validate(row) for key, row in previous.items()}
            for ref, text, *_ in rows:
                if text.strip() == "Two reviewers approve US releases.":
                    states["WRK-0000"].status = "supported"
                    states["WRK-0000"].primary_ref = ref
            carried_ref = previous["WRK-0000"]["primary_ref"]
            if carried_ref and carried_ref not in {row[0] for row in rows}:
                states["WRK-0001"].status = "supported"
                states["WRK-0001"].primary_ref = carried_ref
                self.borrowed = True
            for state in states.values():
                state.reason = "The same approval rule supports both fixed claims"
            return SupportAssessmentResponse(results=list(states.values()))

    client = SharedEvidenceClient()
    items = work_items("Two reviewers approve US releases.\n\n" + "\n\n".join(f"Routine note {i}." for i in range(300)), 2)
    executor = RevisionWorkExecutor(client=client, model="fixture")
    results = await executor.assess_many(items)
    assert client.borrowed and len(client.prompts) > 1
    assert all(result.supported is True for result in results.values())
    assert all(result.memory is not None for result in results.values())
    assert not any("Correction:" in prompt for prompt in client.prompts)
