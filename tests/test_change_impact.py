"""Change Impact routing, request shape, OR merge and failure handling; fixture labels are not model-accuracy evidence."""

import json
import logging
from dataclasses import replace

import pytest
from pydantic import ValidationError

from memforge.llm.structured import (
    ChangeImpactWireResponse,
    LiteLlmStructuredClient,
    StructuredLlmConfig,
    StructuredLlmError,
)
from memforge.memory.evidence import EvidenceRole
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.revision_work import (
    CHANGE_IMPACT_CONTRACT,
    CHANGE_IMPACT_PROMPT,
    SUPPORT_ASSESSMENT_CONTRACT,
    RevisionWorkExecutor,
)
from memforge.pipeline.support_reading import SupportRoute, SupportWorkItem, plan_support_revision
from tests.revision_client_fixture import change_impact_payload
from tests.test_revision_assessment import memory, revisions
from tests.test_revision_work import CHANGED, RULE, Client, Store, payload, receipts, work_items
from tests.test_support_reading import part

GLOBAL_SCOPE_RULE = '"this document" or "discontinued from a given date", is affected.'


class ImpactClient(Client):
    """Label each Change Impact work with ``label(work, data)``; Support reading follows ``Client``."""

    def __init__(self, label=lambda work, data: "unaffected", limit=100000):
        super().__init__(limit=limit)
        self.label = label

    def impact(self, work, data):
        return self.label(work, data)


def current_texts(data):
    return {row[0]: row[1] for row in data["current"]["fragments"]}


def changed_texts(data):
    texts = current_texts(data)
    return sorted(texts[ref] for ref in data["changed_refs"])


def cohort(old, new, supports):
    """Work items that share one base/target pair; ``supports`` maps work id to (claim, Evidence texts)."""
    base, current = revisions(old, new)
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    return [
        SupportWorkItem(
            work_id,
            replace(memory(), id=f"memory-{work_id}", content=claim),
            tuple(
                part(base, text, role=EvidenceRole.PRIMARY if index == 0 else EvidenceRole.REQUIRED)
                for index, text in enumerate(texts)
            ),
            context,
        )
        for work_id, (claim, texts) in supports.items()
    ]


def impact_works(store):
    return [work for work in store.works.values() if work.kind == "change_impact"]


@pytest.mark.asyncio
async def test_appended_minutes_rebind_unrelated_claims_without_support_call():
    decisions = [
        "Decision: Alder ships in October.",
        "Decision: Birch keeps two reviewers.",
        "Decision: Cedar moves to the new queue.",
    ]
    old = "# Meeting 2026-09-01\n\n" + "\n\n".join(decisions) + "\n"
    new = old + "\n# Meeting 2026-09-08\n\nThe team reviewed dashboards.\n"
    items = cohort(old, new, {f"w{i}": (decision.removeprefix("Decision: "), [decision]) for i, decision in enumerate(decisions)})
    client, store = ImpactClient(), Store()
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    results = await executor.assess_many(items)

    [request] = [change_impact_payload(p) for p in client.impact_prompts]
    assert [work["work_id"] for work in request["works"]] == ["WRK-0000", "WRK-0001", "WRK-0002"]
    assert [work["evidence"] for work in request["works"]] == [
        [{"role": "primary", "text": decision, "heading_context": ["# Meeting 2026-09-01"]}] for decision in decisions
    ]
    # Only the appended section changed; no ref or prose is asked for.
    assert changed_texts(request) == ["# Meeting 2026-09-08", "The team reviewed dashboards."]
    assert not any(decision in current_texts(request).values() for decision in decisions)
    assert client.prompts == []
    assert all(r.supported and r.memory.support_validation["route"] == "change_impact" for r in results.values())
    assert all(r.memory.support_validation["model"] == "fixture" for r in results.values())
    assert [
        [p.anchor.observation_revision_id for p in r.memory.resolved_evidence_selection.parts] for r in results.values()
    ] == [["rev-primary-v2"]] * 3
    [impact] = impact_works(store)
    [receipt] = receipts(store)
    assert receipt.manifest["contract"] == CHANGE_IMPACT_CONTRACT and receipt.manifest["completion"] == "program"
    assert receipt.manifest["dependencies"] == [[impact.id, impact.result_hash]]
    assert receipt.result == {"results": [{"work_id": f"w{i}", "impact": "unaffected"} for i in range(3)]}
    assert executor.final_work_ids == [receipt.id]
    assert executor.change_impact_counts == {"unaffected": 3, "affected": 0, "failed": 0}
    assert executor.program_rebind_count == 0 and executor.stage_counts == {"support_assess": 0, "change_impact": 1}


@pytest.mark.asyncio
async def test_punctuation_change_assesses_only_its_own_support():
    old = "# A\n\nRule A holds.\n\n# B\n\nRule B holds.\n\n# C\n\nRule C holds.\n"
    new = old.replace("Rule A holds.", "Rule A holds!")
    items = cohort(old, new, {
        "a": ("A holds.", ["Rule A holds."]), "b": ("B holds.", ["Rule B holds."]), "c": ("C holds.", ["Rule C holds."]),
    })
    routes = {support.item.id: support.route for support in plan_support_revision(items[0].context, items).supports}
    assert routes == {"a": SupportRoute.SUPPORT_ASSESSMENT, "b": SupportRoute.CHANGE_IMPACT, "c": SupportRoute.CHANGE_IMPACT}
    client = ImpactClient()
    results = await RevisionWorkExecutor(client=client, model="fixture").assess_many(items)

    assert {work["work_id"] for p in client.impact_prompts for work in change_impact_payload(p)["works"]} == {
        "WRK-0001", "WRK-0002",
    }
    assert {work["work_id"] for p in client.prompts for work in payload(p)["works"]} == {"WRK-0000"}
    # The fixture finds no support for the modified rule, so it is unsupported only after the whole order.
    assert results["a"].supported is False
    assert all(results[key].memory.support_validation["route"] == "change_impact" for key in ("b", "c"))


@pytest.mark.asyncio
async def test_removed_distant_qualifier_is_affected_and_assessed():
    qualifier = "The rules above apply only to Cedar."
    old = f"# Release\n\n{RULE}\n\n# Scope\n\n{qualifier}\n"
    [item] = cohort(old, f"# Release\n\n{RULE}\n", {"w0": ("US releases require two reviewers.", [RULE])})
    client, store = ImpactClient(
        label=lambda work, data: "affected" if any(qualifier in row[1] for row in data["removed_historical"])
        else "unaffected",
    ), Store()
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    result = (await executor.assess_many([item]))["w0"]

    [request] = [change_impact_payload(p) for p in client.impact_prompts]
    assert qualifier in [row[1] for row in request["removed_historical"]]
    assert all(row[0].startswith("HIS-") for row in request["removed_historical"])
    # The AFFECTED Support reads the same removed text first.
    assert qualifier in [row[1] for row in payload(client.prompts[0])["removed_historical"]]
    assert result.memory.support_validation["route"] == "support_assessment"
    assert executor.change_impact_counts == {"unaffected": 0, "affected": 1, "failed": 0}
    assert {receipt.manifest["contract"] for receipt in receipts(store)} == {SUPPORT_ASSESSMENT_CONTRACT}


@pytest.mark.asyncio
async def test_removed_list_item_brings_its_list_and_headings_into_the_bundle():
    old = (
        f"# Release\n\n{RULE}\n\n# Excluded regions\n\nThe rule above excludes:\n\n- EU\n- APAC\n- LATAM\n"
        "\n# Notes\n\nNote A.\n\nNote B.\n"
    )
    new = old.replace("- APAC\n", "").replace("\n\nNote B.", "")
    [item] = cohort(old, new, {"w0": ("US releases require two reviewers.", [RULE])})
    client = ImpactClient()
    await RevisionWorkExecutor(client=client, model="fixture").assess_many([item])

    [request] = [change_impact_payload(p) for p in client.impact_prompts]
    # The shortened list is a changed ReadingGroup: read whole, with its lead-in, though none of its items changed.
    assert list(current_texts(request).values()) == ["# Excluded regions", "The rule above excludes:", "- EU", "- LATAM"]
    assert request["changed_refs"] == []
    # Removed text keeps the heading path it sat under in the baseline.
    assert [row[1:] for row in request["removed_historical"]] == [
        ["- APAC", {"heading_context": ["# Excluded regions"]}],
        ["Note B.", {"heading_context": ["# Notes"]}],
    ]
    assert "removed_observations" not in request and "observations" not in request["current"]


@pytest.mark.asyncio
async def test_global_scope_statement_is_affected():
    # The rule is fixed by the prompt; it has no dedicated model evaluation cases.
    statement = "This document is discontinued from 2026-10-01."
    [item] = work_items(f"{RULE}\n\n{statement}\n")
    client = ImpactClient(
        label=lambda work, data: "affected" if statement in changed_texts(data) else "unaffected",
    )
    result = (await RevisionWorkExecutor(client=client, model="fixture").assess_many([item]))["w0"]

    [prompt] = client.impact_prompts
    assert GLOBAL_SCOPE_RULE in prompt and GLOBAL_SCOPE_RULE in CHANGE_IMPACT_PROMPT
    assert statement in changed_texts(change_impact_payload(prompt))
    assert client.prompts and result.memory.support_validation["route"] == "support_assessment"


@pytest.mark.asyncio
async def test_change_impact_failure_routes_to_support_assessment_without_a_label(caplog):
    class FailingImpactClient(ImpactClient):
        async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
            if response_format is not ChangeImpactWireResponse:
                return await super().evaluate_revision_work(prompt, response_format=response_format, **kwargs)
            self.impact_prompts.append(prompt)
            raise StructuredLlmError("fixture outage", terminal_category="provider_error", error_code="provider_error")

    client, store = FailingImpactClient(), Store()
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    with caplog.at_level(logging.WARNING, logger="memforge.pipeline.revision_work"):
        results = await executor.assess_many(work_items(CHANGED, 2))

    # Both works shared the failed request, so both are read; neither is rebound.
    assert all(r.supported and r.memory.support_validation["route"] == "support_assessment" for r in results.values())
    assert {work["work_id"] for p in client.prompts for work in payload(p)["works"]} == {"WRK-0000", "WRK-0001"}
    assert executor.change_impact_counts == {"unaffected": 0, "affected": 0, "failed": 2}
    assert len(client.impact_prompts) == 1
    # The failure is journaled as retryable work, and no label is recorded anywhere.
    assert [work.status for work in impact_works(store)] == ["retryable_failure"]
    assert all(work.result is None for work in impact_works(store))
    assert "impact" not in json.dumps([receipt.result for receipt in receipts(store)])
    records = [r.getMessage() for r in caplog.records if r.getMessage().startswith("change_impact_failed")]
    assert len(records) == 2 and all("memory_id=memory-" in record for record in records)
    assert all("category=provider_error" in record for record in records)


@pytest.mark.asyncio
async def test_change_impact_output_that_stays_invalid_isolates_one_work_and_routes_it_to_support_assessment(caplog):
    class OmittingImpactClient(ImpactClient):
        async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
            if response_format is not ChangeImpactWireResponse:
                return await super().evaluate_revision_work(prompt, response_format=response_format, **kwargs)
            self.impact_prompts.append(prompt)
            works = change_impact_payload(prompt)["works"]
            # WRK-0001 is never answered, even after the correction.
            return ChangeImpactWireResponse.model_validate({"results": [
                {"work_id": work["work_id"], "impact": "unaffected"} for work in works if work["work_id"] != "WRK-0001"
            ]})

    client, store = OmittingImpactClient(), Store()
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    with caplog.at_level(logging.WARNING, logger="memforge.pipeline.revision_work"):
        results = await executor.assess_many(work_items(CHANGED, 2))

    # WRK-0000 is judged alone and rebound; only WRK-0001 is read by Support Assessment.
    assert executor.change_impact_counts == {"unaffected": 1, "affected": 0, "failed": 1}
    assert results["w0"].rebound and not results["w1"].rebound
    assert results["w1"].memory.support_validation["route"] == "support_assessment"
    assert {work["work_id"] for p in client.prompts for work in payload(p)["works"]} == {"WRK-0001"}
    # Both works with the correction, then WRK-0000 alone, then WRK-0001 alone with its correction.
    assert len(client.impact_prompts) == 5
    records = [r.getMessage() for r in caplog.records if r.getMessage().startswith("change_impact_failed")]
    assert len(records) == 1 and "memory_id=memory-1" in records[0] and "category=invalid_response" in records[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger_chunk", [True, False])
async def test_chunked_bundle_is_or_merged(trigger_chunk):
    trigger = "Trigger: releases now need one reviewer."
    steps = "\n".join(f"- Cedar step {i}" for i in range(5))
    notes = [f"Appended routine note {i}: maintain ordinary settings." for i in range(80)]
    appended = [*notes[:25], steps, *notes[25:50], *([trigger] if trigger_chunk else []), *notes[50:]]
    [item] = work_items(f"{RULE}\n\n" + "\n\n".join(appended) + "\n")
    # Room for a Support reading step, but not for the claim with every change at once.
    client, store = ImpactClient(
        label=lambda work, data: "affected" if trigger in current_texts(data).values() else "unaffected", limit=4500,
    ), Store()
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    result = (await executor.assess_many([item]))["w0"]

    requests = [change_impact_payload(p) for p in client.impact_prompts]
    assert len(requests) > 1 and all(len(data["works"]) == 1 for data in requests)
    # Chunks follow ReadingGroup boundaries: the list is never split across requests.
    step_counts = [sum("Cedar step" in text for text in current_texts(data).values()) for data in requests]
    assert sorted(step_counts)[-1] == 5 and set(step_counts) <= {0, 5}
    assert sum(any(note in current_texts(data).values() for data in requests) for note in notes) == len(notes)
    if trigger_chunk:
        assert sum(trigger in current_texts(data).values() for data in requests) == 1
        assert result.memory.support_validation["route"] == "support_assessment" and client.prompts
        assert executor.change_impact_counts["affected"] == 1
        return
    assert result.memory.support_validation["route"] == "change_impact" and not client.prompts
    [receipt] = receipts(store)
    assert receipt.manifest["contract"] == CHANGE_IMPACT_CONTRACT
    assert sorted(receipt.manifest["dependencies"]) == sorted(
        [work.id, work.result_hash] for work in impact_works(store)
    )
    assert len(receipt.manifest["dependencies"]) == len(requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["omitted", "unknown"])
async def test_work_id_coverage_is_validated(defect):
    class OnceInvalidClient(ImpactClient):
        async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
            response = await super().evaluate_revision_work(prompt, response_format=response_format, **kwargs)
            if response_format is not ChangeImpactWireResponse or "<correction>" in prompt:
                return response
            rows = [row.model_dump() for row in response.results]
            rows[-1] = {**rows[-1], "work_id": "WRK-9999"} if defect == "unknown" else rows[-1]
            return ChangeImpactWireResponse.model_validate(
                {"results": rows[:-1] if defect == "omitted" else rows}
            )

    client = OnceInvalidClient()
    executor = RevisionWorkExecutor(client=client, model="fixture")
    results = await executor.assess_many(work_items(CHANGED, 2))
    assert len(client.impact_prompts) == 2 and "<correction>" in client.impact_prompts[1]
    # The correction completes the same request instead of sending a new one.
    assert executor.stage_counts["change_impact"] == 1
    # The corrected response decides both works.
    assert all(r.memory.support_validation["route"] == "change_impact" for r in results.values())
    assert executor.change_impact_counts == {"unaffected": 2, "affected": 0, "failed": 0}


@pytest.mark.asyncio
async def test_supports_share_one_bundle_request_and_retry_reuses_it():
    other = "Hotfixes need one reviewer."
    old = f"{RULE}\n\n{other}\n"
    items = cohort(old, old + "\nNew unrelated note.\n", {
        "pair": ("US releases require two reviewers.", [RULE, "Country: US."]),
        "other": (other, [other]),
    })
    client, store = ImpactClient(), Store()
    first = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    results = await first.assess_many(items)

    [request] = [change_impact_payload(p) for p in client.impact_prompts]
    # One question per whole Support: the Primary and Required parts travel in one work.
    assert [[row["role"] for row in work["evidence"]] for work in request["works"]] == [["primary", "required"], ["primary"]]
    assert [row["text"] for row in request["works"][0]["evidence"]] == [RULE, "Country: US."]
    assert [p.excerpt for p in results["pair"].memory.resolved_evidence_selection.parts] == [RULE, "Country: US."]

    retry_client = ImpactClient()
    retry = RevisionWorkExecutor(client=retry_client, model="fixture", store=store, derivation_id="root")
    retried = await retry.assess_many(items)
    assert retry_client.impact_prompts == [] and retry.reused == 1 and retry.calls == 0
    assert retry.final_work_ids == first.final_work_ids
    assert {key: r.memory.resolved_evidence_selection for key, r in retried.items()} == {
        key: r.memory.resolved_evidence_selection for key, r in results.items()
    }


def test_wire_row_is_only_a_work_id_and_a_label_for_strict_transport():
    schema = ChangeImpactWireResponse.model_json_schema()
    row = schema["$defs"]["ChangeImpactWireResult"]
    assert set(row["properties"]) == set(row["required"]) == {"work_id", "impact"}
    assert row["additionalProperties"] is False
    assert row["properties"]["impact"]["enum"] == ["affected", "unaffected"]
    for invalid in ({"impact": "unknown"}, {"reason": "The note is unrelated."}):
        with pytest.raises(ValidationError):
            ChangeImpactWireResponse.model_validate(
                {"results": [{"work_id": "WRK-0000", "impact": "unaffected", **invalid}]}
            )


@pytest.mark.asyncio
async def test_refused_change_impact_reply_is_a_failure_not_a_label(monkeypatch):
    from tests.test_structured_llm import CompletionResponse, set_native_schema_support

    response = CompletionResponse(json.dumps({"results": [{"work_id": "WRK-0000", "impact": "unaffected"}]}))
    response.choices[0].message.refusal = "refused"

    async def completion(**kwargs):
        return response

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", completion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(StructuredLlmConfig(
        model="gpt-4o", base_url=None, api_key=None, timeout_s=10, num_retries=0))
    with pytest.raises(StructuredLlmError) as raised:
        await client.evaluate_revision_work("fixture", response_format=ChangeImpactWireResponse, max_tokens=1024)
    assert raised.value.error_code == "change_impact_response_incomplete"
