import json
from dataclasses import replace

import pytest

from memforge.pipeline.revision_work import (
    RevisionWorkExecutor, SupportWorkItem, ScanResponse, ScanResult, Finding,
    FinalResponse, FinalResult,
)
from tests.test_revision_assessment import revisions, old_support, memory


class Client:
    def __init__(self, limit=4000):
        self.limit = limit
        self.prompts = []
        self.fail_next_scan = False

    def request_fits(self, prompt, *, max_tokens, **kwargs):
        return len(prompt) + max_tokens // 8 <= self.limit

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
            findings = [Finding(kind="scope" if "Cedar" in row[1] else "support", refs=[row[0]], explanation=row[1])
                        for row in rows if "reviewers" in row[1] or "Cedar" in row[1]]
            return ScanResponse(results=[ScanResult(work_id=claim["work_id"], observations_found=findings,
                                                     no_local_effect=not findings) for claim in payload["claims"]])
        text = "\n".join(row[1] for row in rows)
        assert "summaries are navigation" in prompt
        unsupported = "Cedar means US" in text and "Cedar uses one reviewer" in text
        primary = next((row[0] for row in rows if "reviewers" in row[1]), None)
        return FinalResponse(results=[FinalResult(work_id=claim["work_id"], status="unsupported" if unsupported or primary is None else "supported",
                                                 primary_ref=primary, reason="fixture joint decision") for claim in payload["claims"]])


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
    return [SupportWorkItem(f"w{i}", replace(memory(), id=f"memory-{i}"), old_support(base), context) for i in range(count)]


@pytest.mark.asyncio
async def test_small_source_multiple_claims_share_one_final_call():
    client = Client(limit=15000)
    items = work_items("Two reviewers approve US releases.\n", 3)
    executor = RevisionWorkExecutor(client=client, model="fixture")
    result = await executor.assess_many(items)
    assert len(client.prompts) == 1
    assert len(result) == 3 and all(value.supported for value in result.values())
    assert all(value.memory.resolved_evidence_selection.parts[0].anchor.observation_revision_id == "rev-primary-v2" for value in result.values())


@pytest.mark.asyncio
async def test_large_source_scans_all_chunks_and_jointly_explains_distant_exception():
    text = "Two reviewers approve US releases.\n\nCedar uses one reviewer.\n\n" + "\n\n".join(f"Unrelated section {i}: a routine operational note." for i in range(100)) + "\n\nCedar means US releases.\n"
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
    text = "Two reviewers approve US releases.\n\n" + "\n\n".join(f"Unrelated item {i}: still ordinary explanatory text." for i in range(80))
    items = work_items(text)
    client = Client(); client.fail_next_scan = True
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
