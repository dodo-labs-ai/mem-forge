import json
from dataclasses import replace

import pytest

from memforge.pipeline.revision_work import RevisionWorkExecutor, SupportWorkItem
from memforge.llm.structured import (
    RevisionScanResponse as ScanResponse,
    RevisionScanResult as ScanResult,
    RevisionScanFinding as Finding,
    RevisionFinalResponse as FinalResponse,
    RevisionFinalResult as FinalResult,
)
from tests.test_revision_assessment import revisions, old_support, memory


class Client:
    def __init__(self, limit=4000):
        self.limit = limit
        self.prompts = []
        self.fail_next_scan = False

    def request_fits(self, prompt, *, max_tokens, reserve_correction=True, **kwargs):
        return len(prompt) + max_tokens // 8 <= self.limit - (256 if reserve_correction else 0)

    def request_tokens(self, prompt, **kwargs):
        return len(prompt)

    def input_policy_identity_for(self, model=None):
        return f"fixture-{self.limit}-{model}"

    async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
        self.prompts.append(prompt)
        tag = "scan" if response_format is ScanResponse else "final"
        payload = json.loads(prompt.split(f"<{tag}>")[1].split(f"</{tag}>")[0])
        rows = payload["current"]["primary_candidates"]
        if response_format is ScanResponse:
            if self.fail_next_scan and len(self.prompts) > 1:
                self.fail_next_scan = False
                raise TimeoutError("fixture timeout")
            findings = [
                Finding(kind="scope" if "Cedar" in row[1] else "support", refs=[row[0]], explanation=row[1])
                for row in rows
                if "reviewers" in row[1] or "Cedar" in row[1]
            ]
            return ScanResponse(
                results=[
                    ScanResult(work_id=claim["work_id"], observations_found=findings, no_local_effect=not findings)
                    for claim in payload["claims"]
                ]
            )
        text = "\n".join(row[1] for row in rows)
        assert "summaries are navigation" in prompt
        unsupported = "Cedar means US" in text and "Cedar uses one reviewer" in text
        primary = next((row[0] for row in rows if "reviewers" in row[1]), None)
        return FinalResponse(
            results=[
                FinalResult(
                    work_id=claim["work_id"],
                    status="unsupported" if unsupported or primary is None else "supported",
                    primary_ref=primary,
                    reason="fixture joint decision",
                )
                for claim in payload["claims"]
            ]
        )


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


@pytest.mark.asyncio
async def test_small_source_multiple_claims_share_one_final_call():
    client = Client(limit=15000)
    items = work_items("Two reviewers approve US releases.\n", 3)
    executor = RevisionWorkExecutor(client=client, model="fixture")
    result = await executor.assess_many(items)
    assert len(client.prompts) == 1
    assert len(result) == 3 and all(value.supported for value in result.values())
    assert all(
        value.memory.resolved_evidence_selection.parts[0].anchor.observation_revision_id == "rev-primary-v2"
        for value in result.values()
    )


@pytest.mark.asyncio
async def test_large_source_scans_all_chunks_and_jointly_explains_distant_exception():
    text = (
        "Two reviewers approve US releases.\n\nCedar uses one reviewer.\n\n"
        + "\n\n".join(f"Unrelated section {i}: a routine operational note." for i in range(100))
        + "\n\nCedar means US releases.\n"
    )
    client = Client()
    executor = RevisionWorkExecutor(client=client, model="fixture")
    result = await executor.assess_many(work_items(text, 2))
    assert all(not value.supported for value in result.values())
    assert sum("<scan>" in prompt for prompt in client.prompts) > 2
    assert all(client.request_fits(prompt, max_tokens=512) for prompt in client.prompts)
    for prompt in client.prompts:
        if "<final>" in prompt:
            assert "Cedar uses one reviewer" in prompt and "Cedar means US" in prompt


@pytest.mark.asyncio
async def test_retry_reuses_completed_scan_siblings():
    text = "Two reviewers approve US releases.\n\n" + "\n\n".join(
        f"Unrelated item {i}: still ordinary explanatory text." for i in range(80)
    )
    items = work_items(text)
    client = Client()
    client.fail_next_scan = True
    store = Store()
    first = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    with pytest.raises(TimeoutError):
        await first.assess_many(items)
    completed = {key for key, work in store.works.items() if work.status == "completed"}
    assert completed
    second = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    assert (await second.assess_many(items))["w0"].supported
    assert second.reused == len(completed)
    assert second.final_work_ids


@pytest.mark.asyncio
async def test_incomplete_scan_gets_one_correction_and_never_reaches_final():
    from memforge.pipeline.reconciler import ReconciliationContractError

    class EmptyClient(Client):
        async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
            self.prompts.append(prompt)
            assert response_format is ScanResponse
            payload = json.loads(prompt.split("<scan>")[1].split("</scan>")[0])
            return ScanResponse(results=[ScanResult(work_id=c["work_id"]) for c in payload["claims"]])

    client = EmptyClient()
    executor = RevisionWorkExecutor(client=client, model="fixture")
    text = "Two reviewers approve US releases.\n\n" + "\n\n".join(f"Note {i} has no change." for i in range(200))
    with pytest.raises(ReconciliationContractError, match="correction exhausted"):
        await executor.assess_many(work_items(text))
    assert len(client.prompts) == 2
    assert not executor.final_work_ids


@pytest.mark.asyncio
async def test_final_restores_exact_heading_even_when_scan_only_selects_paragraph():
    text = "# US releases\n\nTwo reviewers approve US releases.\n\n" + "\n\n".join(
        f"Routine note {i}." for i in range(200)
    )
    client = Client()
    result = await RevisionWorkExecutor(client=client, model="fixture").assess_many(work_items(text))
    assert result["w0"].supported
    final_prompt = next(prompt for prompt in client.prompts if "<final>" in prompt)
    payload = json.loads(final_prompt.split("<final>")[1].split("</final>")[0])
    rows = payload["current"]["primary_candidates"] + payload["current"]["required_only_candidates"]
    assert any(row[1].strip() == "# US releases" for row in rows)
    assert all("US releases" not in f["explanation"] or "reviewers" in f["explanation"] for f in payload["findings"])


@pytest.mark.asyncio
async def test_shared_old_evidence_is_sent_once_but_supports_stay_separate():
    client = Client(limit=15000)
    await RevisionWorkExecutor(client=client, model="fixture").assess_many(
        work_items("Two reviewers approve US releases.\n", 3)
    )
    payload = json.loads(client.prompts[0].split("<final>")[1].split("</final>")[0])
    assert len(payload["historical_evidence"]) == 1
    assert len(payload["previous_evidence"]) == 3
    assert len({group["work_id"] for group in payload["previous_evidence"]}) == 3


@pytest.mark.asyncio
async def test_reduction_accounts_for_all_findings_before_final():
    from memforge.llm.structured import RevisionReductionResponse, RevisionReductionDisposition

    class ReductionClient(Client):
        def __init__(self):
            super().__init__(limit=7000)
            self.reduced_ids = []

        async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
            if response_format is RevisionReductionResponse:
                self.prompts.append(prompt)
                payload = json.loads(prompt.split("<reduce>")[1].split("</reduce>")[0])
                rows = payload["current"]["primary_candidates"] + payload["current"]["required_only_candidates"]
                selected = {row[0] for row in rows if "reviewers" in row[1]}
                self.reduced_ids.extend(f["finding_id"] for f in payload["findings"])
                return RevisionReductionResponse(
                    dispositions=[
                        RevisionReductionDisposition(
                            finding_id=f["finding_id"],
                            retained_refs=[ref for ref in f["refs"] if ref in selected],
                            explanation="Other material is an unrelated operational note.",
                        )
                        for f in payload["findings"]
                    ],
                    summary="Only the approval rule supports this claim.",
                    needs_context=[],
                )
            response = await super().evaluate_revision_work(prompt, response_format=response_format, **kwargs)
            if response_format is ScanResponse:
                payload = json.loads(prompt.split("<scan>")[1].split("</scan>")[0])
                for result in response.results:
                    result.observations_found = [
                        Finding(
                            kind="support",
                            refs=[row[0]],
                            explanation="Inspect this potentially relevant operational note. " * 6,
                        )
                        for row in payload["current"]["primary_candidates"]
                    ]
                    result.no_local_effect = False
            return response

    client = ReductionClient()
    text = "Two reviewers approve US releases.\n\n" + "\n\n".join(
        f"Unrelated operational note {i}: maintain the routine settings." for i in range(100)
    )
    result = await RevisionWorkExecutor(client=client, model="fixture").assess_many(work_items(text))
    assert result["w0"].supported
    assert len(client.reduced_ids) >= 101
    assert "<final>" in client.prompts[-1]


@pytest.mark.asyncio
async def test_large_source_and_many_claims_share_scan_requests():
    client = Client(limit=6500)
    text = "Two reviewers approve US releases.\n\n" + "\n\n".join(
        f"Routine operational guidance {i}." for i in range(200)
    )
    executor = RevisionWorkExecutor(client=client, model="fixture")
    result = await executor.assess_many(work_items(text, 8))
    scans = [json.loads(p.split("<scan>")[1].split("</scan>")[0]) for p in client.prompts if "<scan>" in p]
    assert any(len(scan["claims"]) > 1 for scan in scans)
    assert len(result) == 8 and all(item.supported for item in result.values())
    expected_units = len(executor._range(work_items(text, 8)).catalog.fragments)
    assert executor.covered_source_claim_pairs == expected_units * 8


@pytest.mark.asyncio
async def test_oversized_historical_evidence_uses_complete_current_full_scan():
    from memforge.pipeline.revision_assessment import RevisionAssessmentContext

    old = "Two reviewers approve US releases. " + "Historical explanatory background. " * 500
    new = "Two reviewers approve US releases.\n\n" + "\n\n".join(f"Current routine note {i}." for i in range(200))
    base, current = revisions(old, new)
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    item = SupportWorkItem("w0", memory(), old_support(base), context)
    client = Client(limit=6000)
    result = await RevisionWorkExecutor(client=client, model="fixture").assess_many([item])
    assert result["w0"].supported
    scans = [json.loads(p.split("<scan>")[1].split("</scan>")[0]) for p in client.prompts if "<scan>" in p]
    assert len(scans) > 1
    assert all(scan["input_mode"] == "full" and "historical_evidence" not in scan for scan in scans)
    assert any("Current routine note 199" in p for p in client.prompts)


@pytest.mark.asyncio
async def test_different_baselines_share_current_full_when_total_delta_cost_is_higher():
    from memforge.pipeline.revision_assessment import RevisionAssessmentContext

    current_text = "Two reviewers approve US releases.\n\n" + "\n\n".join(
        f"Current operational note {i}: the process is stable." for i in range(80)
    )
    items = []
    for index in range(3):
        base, current = revisions(
            f"Two reviewers approve US releases.\n\nOld baseline {index}: historical policy.\n", current_text
        )
        primary = replace(base.observation_revisions[0], id=f"old-revision-{index}")
        unit = replace(base.source_unit_revisions[0], id=f"old-unit-{index}",
                       observation_revision_ids=(primary.id, "rev-context"))
        base = replace(base, observation_revisions=(primary, base.observation_revisions[1]),
                       source_unit_revisions=(unit,),
                       deltas=(replace(base.deltas[0], current_unit_revision_id=unit.id),))
        context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
        items.append(SupportWorkItem(f"w{index}", replace(memory(), id=f"m{index}"), old_support(base), context))
    client = Client(limit=6500)
    executor = RevisionWorkExecutor(client=client, model="fixture")
    results = await executor.assess_many(items)
    scans = [json.loads(p.split("<scan>")[1].split("</scan>")[0]) for p in client.prompts if "<scan>" in p]
    assert scans and all(p["input_mode"] == "full" for p in scans)
    assert any(len(p["claims"]) > 1 for p in scans)
    assert len(results) == 3 and all(r.supported for r in results.values())
    for item in items:
        seen = {row[1] for p in scans if any(c["work_id"] == item.id for c in p["claims"])
                for row in p["current"]["primary_candidates"]}
        assert {f.presentation_text for f in item.context.full_fragments} <= seen


def test_validated_baseline_can_be_newer_than_immutable_evidence_provenance():
    from memforge.pipeline.revision_assessment import RevisionAssessmentContext

    old = "Two reviewers approve US releases.\n\n" + "\n\n".join(f"Stable note {i}." for i in range(120))
    base, current = revisions(old, old + "\n\nOne new routine note.\n")
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    part = old_support(base)[0]
    part = replace(part, anchor=replace(part.anchor, observation_revision_id="original-v1"),
                   validation_plan_id="validated-later", validation_unit_revision_id=base.source_unit_revisions[0].id)
    item = SupportWorkItem("w0", memory(), (part,), context)
    executor = RevisionWorkExecutor(client=Client(), model="fixture")
    assert executor._range([item]).mode == "delta"
    changed = replace(item, support=(replace(part, validation_plan_id="different-plan"),))
    assert executor._identity([item]) != executor._identity([changed])


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_ref', ['e0', 'p999999', 'quoted source text ' * 2000, '重复文本' * 20])
async def test_scan_correction_identifies_invalid_ref_without_widening_evidence_scope(bad_ref):
    class RepairClient(Client):
        sent_invalid = False

        def request_fits(self, prompt, *, max_tokens, reserve_correction=True, **kwargs):
            return len(prompt) + max_tokens // 8 <= self.limit - (1024 if reserve_correction else 0)

        async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
            result = await super().evaluate_revision_work(prompt, response_format=response_format, **kwargs)
            if response_format is ScanResponse and not self.sent_invalid:
                self.sent_invalid = True
                result.results[0].observations_found = [Finding(kind='support', refs=[bad_ref] + ([f'{i}' + bad_ref[1:] for i in range(8)] if bad_ref.startswith('重复') else []), explanation='fixture')]
                result.results[0].no_local_effect = False
            elif 'Correction:' in prompt:
                diagnostic = prompt.split('Correction:')[1]
                if not bad_ref.startswith('重复'):
                    assert (bad_ref if len(bad_ref) <= 80 else f'<invalid ref: {len(bad_ref)} chars>') in diagnostic
                assert len(diagnostic) < 1024
                if len(bad_ref) > 80:
                    assert bad_ref not in diagnostic
                assert 'historical_evidence is not selectable' in prompt
            return result

    client = RepairClient(limit=5000)
    text = 'Two reviewers approve US releases.\n\n' + '\n\n'.join(f'Routine process note {i}.' for i in range(150))
    executor = RevisionWorkExecutor(client=client, model='fixture')
    result = await executor.assess_many(work_items(text))
    assert client.sent_invalid and sum('Correction:' in p for p in client.prompts) == 1
    assert result['w0'].supported
    assert all(part.anchor.observation_revision_id != 'rev-primary'
               for part in result['w0'].memory.resolved_evidence_selection.parts)
