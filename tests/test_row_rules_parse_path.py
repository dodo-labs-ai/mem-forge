"""Row rules run on each parsed row, through the real structured client's parse path.

A response whose shape is valid but whose one row breaks a meaning rule costs one
re-ask of that item alone: two model calls, no split, every other row accepted.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import pytest

from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig
from memforge.memory.candidate_admission import admit_candidates
from memforge.memory.relation_classifier import MemoryPair, StructuredMemoryPairClassifier
from memforge.models import Memory, content_hash
from memforge.pipeline.claim_revision import assess_claim_pairs
from tests.llm_fixture import admission_payload
from tests.revision_client_fixture import catalog_payload
from tests.test_candidate_admission import candidate as admission_candidate
from tests.test_claim_revision import candidate as revision_candidate
from tests.test_revision_assessment import memory
from tests.test_structured_llm import CompletionResponse, set_native_schema_support

MODEL = "provider/fixture-model"
# Large enough that every request of these fixtures fits one call.
ROUTE_METADATA = {"max_input_tokens": 200_000, "context_window": 400_000, "max_output_tokens": 32_000}
CORRECTION = "<correction>"


def parse_path_client(monkeypatch, respond: Callable[[str], dict]) -> tuple[LiteLlmStructuredClient, list[str]]:
    """A LiteLLM structured client whose provider returns ``respond(prompt)`` as JSON text."""
    prompts: list[str] = []

    async def completion(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        prompts.append(prompt)
        return CompletionResponse(json.dumps(respond(prompt)))

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", completion)
    monkeypatch.setattr("memforge.llm.request_budget.litellm.get_model_info", lambda _model: ROUTE_METADATA)
    monkeypatch.setattr(
        "memforge.llm.structured.litellm.token_counter",
        lambda *, model, messages: len(messages[0]["content"].split()),
    )
    set_native_schema_support(monkeypatch, False)
    client = LiteLlmStructuredClient(StructuredLlmConfig(
        model=MODEL, base_url=None, api_key=None, timeout_s=5.0, input_budget_fraction=1.0, num_retries=0,
    ))
    return client, prompts


@pytest.mark.asyncio
async def test_one_claim_revision_row_without_its_contradiction_proof_is_re_asked_alone(monkeypatch):
    def respond(prompt):
        rows = []
        for claim in catalog_payload(prompt)["new_claims"]:
            relations = []
            if claim["id"] == "NEW-0002" and CORRECTION not in prompt:
                # Valid shape, invalid meaning: CONTRADICTS without its proof.
                relations = [{"existing_id": "MEM-0001", "relation": "contradicts"}]
            rows.append({"candidate_id": claim["id"], "relations": relations, "uncertain_existing_ids": []})
        return {"results": rows}

    client, prompts = parse_path_client(monkeypatch, respond)
    candidates = [replace(revision_candidate(), content=f"claim {index}") for index in range(3)]

    ledger = await assess_claim_pairs(candidates=candidates, incumbents=[memory()], client=client, model=MODEL)

    assert len(prompts) == 2
    assert [claim["id"] for claim in catalog_payload(prompts[1])["new_claims"]] == ["NEW-0002"]
    assert "NEW-0002: MEM-0001: CONTRADICTS requires overlapping scope" in prompts[1]
    assert ledger.completed_candidate_count == 3 and not ledger.unjudged


@pytest.mark.asyncio
async def test_one_admission_row_rejected_without_a_reason_is_re_asked_alone(monkeypatch):
    def respond(prompt):
        decisions = []
        for row in admission_payload(prompt)["candidates"]:
            decision = {"candidate_id": row["id"], "verdict": "ADMITTED"}
            if row["id"] == "CND-0002" and CORRECTION not in prompt:
                # Valid shape, invalid meaning: REJECTED names no reject_reason.
                decision = {"candidate_id": row["id"], "verdict": "REJECTED"}
            decisions.append(decision)
        return {"decisions": decisions}

    client, prompts = parse_path_client(monkeypatch, respond)
    claims = ["US releases require two reviewers.", "Hotfixes need one reviewer.", "Audit logs are kept."]

    result = await admit_candidates([admission_candidate(claim) for claim in claims], client=client, model=MODEL)

    assert len(prompts) == 2
    assert [row["id"] for row in admission_payload(prompts[1])["candidates"]] == ["CND-0002"]
    assert "CND-0002: REJECTED requires reject_reason" in prompts[1]
    assert len(result.admitted) == 3 and result.rejected == ()


@pytest.mark.asyncio
async def test_one_memory_relation_row_with_a_wrong_direction_is_re_asked_alone(monkeypatch):
    def groups(prompt):
        return json.loads(prompt.split("<memory_pair_groups>\n", 1)[1].split("\n</memory_pair_groups>", 1)[0])

    def respond(prompt):
        decisions = []
        for group in groups(prompt):
            for item in group["candidates"]:
                # Valid shape, invalid meaning: a symmetric REFINES.
                refines = item["pair_index"] == 1 and CORRECTION not in prompt
                decisions.append({
                    "pair_index": item["pair_index"], "classification": "refines" if refines else "unrelated",
                    "direction": "symmetric", "same_subject_and_scope": False, "incompatible_assertions": "",
                })
        return {"decisions": decisions}

    client, prompts = parse_path_client(monkeypatch, respond)

    def claim(memory_id: str) -> Memory:
        text = f"{memory_id} claim"
        return Memory(id=memory_id, content=text, content_hash=content_hash(text), memory_type="fact")

    pairs = tuple(MemoryPair(claim(f"new-{index}"), claim(f"old-{index}")) for index in range(3))

    result = await StructuredMemoryPairClassifier(client=client, model=MODEL).classify(pairs)

    assert len(prompts) == 2
    assert [item["pair_index"] for group in groups(prompts[1]) for item in group["candidates"]] == [1]
    assert "pair_index 1: REFINES must be directional" in prompts[1]
    assert len(result.decisions) == 3 and result.unjudged == ()


@pytest.mark.asyncio
async def test_a_shifted_claim_revision_answer_never_lends_one_candidates_contradiction_to_another(monkeypatch):
    contradiction = {
        "existing_id": "MEM-0001", "relation": "contradicts",
        "contradiction": {"same_subject_and_scope": True, "incompatible_assertions": "two versus one reviewer"},
    }

    def respond(prompt):
        claims = catalog_payload(prompt)["new_claims"]
        if CORRECTION in prompt:
            return {"results": [
                {"candidate_id": claim["id"], "relations": [], "uncertain_existing_ids": []} for claim in claims
            ]}
        # NEW-0001's contradiction is answered under NEW-0002, and so on; the last row names NEW-0004.
        return {"results": [
            {"candidate_id": f"NEW-{index + 2:04d}", "relations": [contradiction] if index == 0 else [],
             "uncertain_existing_ids": []}
            for index, _claim in enumerate(claims)
        ]}

    client, prompts = parse_path_client(monkeypatch, respond)
    candidates = [replace(revision_candidate(), content=f"claim {index}") for index in range(3)]

    ledger = await assess_claim_pairs(candidates=candidates, incumbents=[memory()], client=client, model=MODEL)

    # The unknown NEW-0004 voids the whole response; the correction resends every candidate.
    assert len(prompts) == 2
    assert [claim["id"] for claim in catalog_payload(prompts[1])["new_claims"]] == ["NEW-0001", "NEW-0002", "NEW-0003"]
    assert "NEW-0004" in prompts[1].split(CORRECTION)[1]
    assert ledger.decisions == () and ledger.completed_candidate_count == 3
