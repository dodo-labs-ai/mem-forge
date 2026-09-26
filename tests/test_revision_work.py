"""Ordered Support reading contracts; fixture judgments are not model-accuracy evidence."""

import json
import logging
from dataclasses import replace

import pytest

from memforge.llm.structured import ChangeImpactWireResponse, StructuredLlmError, SupportAssessmentWireResponse
from memforge.memory.evidence import EvidenceRole
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.revision_work import SUPPORT_ASSESSMENT_CONTRACT, RevisionWorkExecutor
from memforge.pipeline.support_reading import SupportWorkItem, plan_support_revision
from tests.revision_client_fixture import change_impact_response, continued, supported, unsupported
from tests.test_revision_assessment import memory, old_support, revisions
from tests.test_support_reading import part

RULE = "Two reviewers approve US releases."
EXCEPTION_TERMS = ("Cedar", "Alder", "Birch")
# A target that changed beside the unchanged rule: the exact Support first goes through
# Change Impact, which the fixture client judges AFFECTED, and is then read.
CHANGED = f"{RULE}\n\nRelease notes are published weekly.\n"


def payload(prompt):
    return json.loads(prompt.split("<assessment>")[1].split("</assessment>")[0])


def is_impact(prompt):
    return "<change_impact>" in prompt


def readable(data):
    """Every (ref, text) the request lets the model read and select."""
    current = data["current"]
    return [
        (row[0], row[1])
        for row in (*current["primary_candidates"], *current["required_only_candidates"], *data["carried_witness_catalog"])
    ]


class Client:
    """Support the release rule unless a complete exception has been read."""

    def __init__(self, limit=8000):
        self.limit = limit
        self.prompts = []
        self.impact_prompts = []
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
        if response_format is ChangeImpactWireResponse:
            self.impact_prompts.append(prompt)
            return change_impact_response(prompt, self.impact)
        assert response_format is SupportAssessmentWireResponse
        self.prompts.append(prompt)
        if self.fail_at == len(self.prompts):
            self.fail_at = None
            raise TimeoutError("fixture interruption")
        return SupportAssessmentWireResponse.model_validate({"results": self.judge(prompt)})

    def impact(self, work, data):
        """Every change may affect the claim, so exact Supports are read like any other."""
        return "affected"

    def judge(self, prompt):
        data = payload(prompt)
        rows = readable(data)
        rule = [ref for ref, text in rows if text.strip() == RULE]
        opposing = [ref for ref, text in rows if any(term in text for term in EXCEPTION_TERMS)]
        combined = " ".join(text for ref, text in rows if ref in opposing)
        exception = ("Cedar uses one reviewer" in combined and "Cedar means US" in combined) or all(
            fact in combined for fact in ("Alder releases need not", "Alder refers to Birch", "Birch refers to US")
        )
        results = []
        for work in data["works"]:
            if not exception and rule and work["may_conclude"]:
                results.append(supported(work, rule[0]))
            elif data["last"]:
                results.append(unsupported(work))
            else:
                results.append(continued(work, rule, opposing))
        return results


class Store:
    def __init__(self):
        self.works = {}

    async def stage_derivation_work(self, *, derivation_id, work):
        return self.works.setdefault((derivation_id, work.id), work)

    async def record_derivation_work(self, *, derivation_id, work):
        self.works[derivation_id, work.id] = work
        return work


def work_items(text, count=1, *, old=f"{RULE}\n"):
    base, current = revisions(old, text)
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    return [
        SupportWorkItem(f"w{i}", replace(memory(), id=f"memory-{i}"), old_support(base), context) for i in range(count)
    ]


def without_baseline(item, support=None):
    context = RevisionAssessmentContext(projection=item.context.projection, base=None, access_context_hash="scope")
    return replace(item, context=context, support=item.support if support is None else support)


def receipts(store):
    return [w for w in store.works.values() if w.kind == "support_finalize"]


def reading_order(items):
    """The reading order of a cohort in which every Support is read."""
    revision_plan = plan_support_revision(items[0].context, items)
    return revision_plan.reading_order(revision_plan.supports)


def selected_texts(result):
    return [p.excerpt for p in result.memory.resolved_evidence_selection.parts]


@pytest.mark.asyncio
async def test_small_change_shares_one_request_and_program_completion():
    client, store = Client(limit=16000), Store()
    items = work_items(CHANGED, 3)
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    results = await executor.assess_many(items)
    assert len(client.prompts) == 1 and len(results) == 3
    assert all(r.supported for r in results.values())
    assert set(payload(client.prompts[0])["works"][0]) == {
        "work_id", "claim", "memory_type", "valid_from", "valid_until", "prior_evidence", "previous_state", "may_conclude",
    }
    [receipt] = receipts(store)
    assert receipt.manifest["completion"] == "program" and receipt.manifest["contract"] == SUPPORT_ASSESSMENT_CONTRACT
    # One Change Impact request judged the three exact Supports AFFECTED; one reading request followed.
    assert receipt.manifest["dependencies"] and len(client.impact_prompts) == 1 and executor.calls == 2
    assert all(
        r.memory.resolved_evidence_selection.parts[0].anchor.observation_revision_id == "rev-primary-v2"
        for r in results.values()
    )
    assert all(r.memory.support_validation["route"] == "support_assessment" for r in results.values())


@pytest.mark.asyncio
async def test_appended_section_exits_after_the_first_part_without_reading_the_rest():
    body = f"{RULE}\n\n" + "\n\n".join(f"Unchanged unrelated note {i}." for i in range(100))
    [item] = work_items(body + "\n\nNew unrelated execution record.", old=body)
    client, store = Client(limit=100000), Store()
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    [result] = (await executor.assess_many([item])).values()
    # Everything would fit one request, but the Support concludes after the appended note and its own rule.
    assert len(client.prompts) == 1
    assert "New unrelated execution record" in client.prompts[0]
    assert "Unchanged unrelated note 99" not in client.prompts[0]
    assert result.supported and selected_texts(result) == [RULE]
    [receipt] = receipts(store)
    assert receipt.manifest["coverage"]["read_parts"] == {"w0": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_cross_request_exception_survives_later_unrelated_and_rule_text(reverse):
    pieces = ["Cedar uses one reviewer.", "Cedar means US releases.", RULE]
    if reverse:
        pieces.reverse()
    filler = "\n\n".join(f"Routine execution note {i}: maintain settings." for i in range(100))
    text = ("\n\n" + filler + "\n\n").join(pieces)
    client, store = Client(), Store()
    items = work_items(text, 3)
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    results = await executor.assess_many(items)
    assert all(r.supported is False for r in results.values())
    assert len(client.prompts) > 2
    assert any(len(payload(p)["works"]) > 1 for p in client.prompts)
    # The exception pieces were read in different requests and carried with their exact text.
    assert any(
        any("Cedar" in row[1] for row in payload(p)["carried_witness_catalog"]) for p in client.prompts
    )
    # Unsupported only after the whole order was read.
    total = len(reading_order(items).parts)
    assert all(
        count == total for receipt in receipts(store) for count in receipt.manifest["coverage"]["read_parts"].values()
    )


@pytest.mark.asyncio
async def test_retry_reuses_completed_assessments_without_replaying_their_model_calls():
    items = work_items(f"{RULE}\n\n" + "\n\n".join(f"Routine note {i}." for i in range(350)))
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
async def test_unknown_selector_gets_one_local_correction_not_a_completed_receipt(caplog):
    class InvalidClient(Client):
        def judge(self, prompt):
            results = super().judge(prompt)
            for row in results:
                if row["status"] == "supported":
                    row["primary_ref"] = "not-supplied"
            return results

    client = InvalidClient()
    executor = RevisionWorkExecutor(client=client, model="fixture")
    with caplog.at_level(logging.WARNING, logger="memforge.pipeline.revision_work"):
        [result] = (await executor.assess_many(work_items(CHANGED))).values()
    assert len(client.prompts) == 2 and "<correction>" in client.prompts[-1]
    # The claim alone stays invalid after its correction: UNRESOLVED(invalid_response) keeps it unchanged.
    assert (result.supported, result.unresolved, result.memory) == (None, "invalid_response", None)
    assert not executor.final_work_ids
    [record] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("support_unresolved_invalid_response")]
    assert "memory_id=memory-0" in record and "error_code=output_invalid" in record


@pytest.mark.asyncio
async def test_one_claim_whose_selection_stays_invalid_leaves_the_others_judged():
    class OneInvalidClient(Client):
        def judge(self, prompt):
            results = super().judge(prompt)
            for row in results:
                if row["work_id"] == "WRK-0001" and row["status"] == "supported":
                    row["primary_ref"] = "not-supplied"
            return results

    client = OneInvalidClient()
    results = await RevisionWorkExecutor(client=client, model="fixture").assess_many(work_items(CHANGED, 2))

    assert results["w0"].supported is True
    assert (results["w1"].supported, results["w1"].unresolved) == (None, "invalid_response")
    # Both claims with the correction, then each claim alone; the invalid one with its correction.
    assert [sorted(work["work_id"] for work in payload(prompt)["works"]) for prompt in client.prompts] == [
        ["WRK-0000", "WRK-0001"], ["WRK-0000", "WRK-0001"], ["WRK-0000"], ["WRK-0001"], ["WRK-0001"],
    ]


@pytest.mark.asyncio
async def test_unusable_baseline_reads_the_whole_revision_before_concluding():
    class EagerClient(Client):
        """Claims support as soon as the rule is readable, even before it may conclude."""

        def judge(self, prompt):
            data = payload(prompt)
            rule = [ref for ref, text in readable(data) if text.strip() == RULE]
            return [supported(work, rule[0]) if rule else continued(work) for work in data["works"]]

    [item] = work_items(f"{RULE}\n\n" + "\n\n".join(f"Routine note {i}." for i in range(300)))
    client = EagerClient()
    result = await RevisionWorkExecutor(client=client, model="fixture").assess_many([without_baseline(item)])
    assert result["w0"].supported and len(client.prompts) > 1
    # Without a usable baseline the first part is the whole revision.
    assert [payload(p)["works"][0]["may_conclude"] for p in client.prompts] == [False] * (len(client.prompts) - 1) + [True]
    assert all(payload(p)["removed_historical"] == [] for p in client.prompts)


def test_output_allowance_is_requested_in_full_and_bounded_only_by_the_route():
    items = work_items(RULE, 32)
    executor = RevisionWorkExecutor(client=Client(), model="gpt-4o")
    assert executor._output([item.id for item in items], 300) > executor.client.request_budget().output_limit


@pytest.mark.asyncio
async def test_growing_witness_state_splits_only_the_unread_tail_and_resumes():
    class GrowingClient(Client):
        def request_fits(self, prompt, **kwargs):
            if is_impact(prompt):
                return super().request_fits(prompt, **kwargs)
            data = payload(prompt)
            witnesses = [w["previous_state"]["support_witness_refs"] for w in data["works"]]
            if len(witnesses) > 1 and any(witnesses):
                return False
            if len(data["current"]["primary_candidates"]) > 12:
                return False
            return super().request_fits(prompt, **kwargs)

        def judge(self, prompt):
            data = payload(prompt)
            if data["last"]:
                return super().judge(prompt)
            refs = [ref for ref, _ in readable(payload(prompt))]
            return [continued(work, refs) for work in data["works"]]

    items = work_items(f"{RULE}\n\n" + "\n\n".join(f"Routine note {i}." for i in range(70)), 2)
    store, client = Store(), GrowingClient(limit=20000)
    client.fail_at = 3
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    with pytest.raises(TimeoutError):
        await executor.assess_many(items)
    assert len(payload(client.prompts[0])["works"]) == 2
    retry_client = GrowingClient(limit=20000)
    retry = RevisionWorkExecutor(client=retry_client, model="fixture", store=store, derivation_id="root")
    assert all(r.supported for r in (await retry.assess_many(items)).values())
    assert retry.reused >= 2
    assert all(len(payload(p)["works"]) == 1 for p in client.prompts[1:] + retry_client.prompts)
    finals = receipts(store)
    assert len(finals) == 2
    assert finals[0].manifest["dependencies"][0] == finals[1].manifest["dependencies"][0]


@pytest.mark.asyncio
async def test_removed_text_is_read_in_the_first_part_but_never_selectable():
    previous = f"{RULE}\n\n" + "\n\n".join(f"Deleted condition {i}." for i in range(250))
    items = work_items(RULE, 3, old=previous)
    client, store = Client(limit=100000), Store()
    executor = RevisionWorkExecutor(client=client, model="gpt-4o", store=store, derivation_id="root")
    results = await executor.assess_many(items)
    assert all(result.supported for result in results.values())
    removed = [row for p in client.prompts for row in payload(p)["removed_historical"]]
    assert len(removed) == 250 and all(row[0].startswith("HIS-") for row in removed)
    assert all(selected_texts(result) == [RULE] for result in results.values())


@pytest.mark.asyncio
async def test_prior_evidence_follows_exact_correspondence_and_history_stays_in_the_first_part():
    other = "Hotfixes need one reviewer."
    notes = "\n\n".join(f"Routine note {i}." for i in range(80))
    base, current = revisions(f"{RULE}\n\n{other}\n\n{notes}\n", f"{RULE[:-1]}!\n\n{other}\n\n{notes}\n")
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    items = [
        SupportWorkItem("changed", memory(), (part(base, RULE),), context),
        SupportWorkItem("exact", replace(memory(), id="exact", content="Hotfixes need one reviewer."),
                        (part(base, other),), context),
    ]

    class NeverClient(Client):
        def judge(self, prompt):
            data = payload(prompt)
            return [unsupported(w) if data["last"] else continued(w) for w in data["works"]]

    client = NeverClient(limit=4500)
    results = await RevisionWorkExecutor(client=client, model="fixture").assess_many(items)
    assert all(result.supported is False for result in results.values())
    requests = [payload(p) for p in client.prompts]
    assert len(requests) > 2
    by_work = {}
    for data in requests:
        for work in data["works"]:
            by_work.setdefault(work["work_id"], []).append(work["prior_evidence"])
    # The punctuation change is MODIFIED: its old text is sent, but only while reading the first part.
    assert by_work["WRK-0000"][0] == [{"role": "primary", "historical_excerpt": RULE}]
    assert all(prior == [] for prior in by_work["WRK-0000"][1:])
    # The other Support is exactly unchanged: only its current ref, in every request.
    assert all(len(prior) == 1 and prior[0]["current_ref"].startswith("PRM-") for prior in by_work["WRK-0001"])
    assert not any(RULE in json.dumps(prior) for prior in by_work["WRK-0001"])


@pytest.mark.asyncio
async def test_deleted_sentence_is_supported_by_distant_text_in_the_second_part():
    distant = "Far appendix: two reviewers approve every US release."
    notes = "\n\n".join(f"Routine note {i}." for i in range(80))
    [item] = work_items(f"{notes}\n\n{distant}\n", old=f"{RULE}\n\n{notes}\n\n{distant}\n")

    class DistantClient(Client):
        def judge(self, prompt):
            data = payload(prompt)
            found = [ref for ref, text in readable(data) if text == distant]
            return [
                supported(w, found[0]) if found and w["may_conclude"]
                else unsupported(w) if data["last"] else continued(w, found)
                for w in data["works"]
            ]

    client = DistantClient(limit=4000)
    [result] = (await RevisionWorkExecutor(client=client, model="fixture").assess_many([item])).values()
    assert result.supported and selected_texts(result) == [distant]
    # The first part is only the removed sentence; the distant text is read later.
    assert payload(client.prompts[0])["removed_historical"][0][1] == RULE
    assert distant not in client.prompts[0] and len(client.prompts) > 1


@pytest.mark.asyncio
async def test_section_deletion_is_unsupported_only_after_the_whole_order():
    notes = "\n\n".join(f"Other note {i}." for i in range(60))
    [item] = work_items(f"# Other\n\n{notes}\n", old=f"# Release\n\n{RULE}\n\n# Other\n\n{notes}\n")

    class DeniedClient(Client):
        def judge(self, prompt):
            # Tries to conclude UNSUPPORTED early; the program ignores it until the last group.
            return [unsupported(w) for w in payload(prompt)["works"]]

    client, store = DeniedClient(limit=4500), Store()
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    [result] = (await executor.assess_many([item])).values()
    assert result.supported is False and result.unresolved is None and result.complete_read
    assert len(client.prompts) > 1
    total = len(reading_order([item]).parts)
    [receipt] = receipts(store)
    assert receipt.manifest["coverage"] == {"total": total, "read_parts": {"w0": total}}
    assert all(any(f"Other note {i}." in p for p in client.prompts) for i in range(60))


@pytest.mark.asyncio
async def test_witness_union_is_monotonic_and_rehydrated():
    class OnceClient(Client):
        """Reports the rule once, then only empty deltas; concludes from the carried witness."""

        def judge(self, prompt):
            data = payload(prompt)
            current = {row[0] for row in data["current"]["primary_candidates"]}
            rule = [ref for ref, text in readable(data) if text.strip() == RULE]
            return [
                supported(w, rule[0]) if data["last"] else continued(w, [r for r in rule if r in current])
                for w in data["works"]
            ]

    [item] = work_items(f"{RULE}\n\n" + "\n\n".join(f"Routine note {i}." for i in range(120)))
    # A legacy part has no digest, so the rule is never an exact prior match: only the witness carries it.
    item = without_baseline(item, support=(replace(item.support[0], raw_content_sha256=None, presentation_sha256=None),))
    client = OnceClient(limit=4500)
    [result] = (await RevisionWorkExecutor(client=client, model="fixture").assess_many([item])).values()
    assert result.supported and selected_texts(result) == [RULE]
    requests = [payload(p) for p in client.prompts]
    assert len(requests) > 2
    first_ref = next(row[0] for row in requests[0]["current"]["primary_candidates"] if row[1] == RULE)
    for data in requests[1:]:
        assert data["works"][0]["previous_state"]["support_witness_refs"] == [first_ref]
        assert [first_ref, RULE] in [list(row[:2]) for row in data["carried_witness_catalog"]]


@pytest.mark.asyncio
async def test_results_do_not_depend_on_split_points():
    [item] = work_items(f"{RULE}\n\n" + "\n\n".join(f"Routine note {i}." for i in range(150)))
    item = without_baseline(item)
    outcomes = []
    for limit in (5000, 8000, 100000):
        client = Client(limit=limit)
        [result] = (await RevisionWorkExecutor(client=client, model="fixture").assess_many([item])).values()
        outcomes.append((result.supported, selected_texts(result), len(client.prompts)))
    assert [outcome[:2] for outcome in outcomes] == [(True, [RULE])] * 3
    requests = [outcome[2] for outcome in outcomes]
    assert requests[0] > requests[1] > requests[2] == 1


@pytest.mark.asyncio
async def test_supports_with_different_first_parts_share_one_request_when_they_fit():
    rules = [f"Service {i} needs two reviewers." for i in range(20)]
    old = "\n\n".join(rules) + "\n"
    base, current = revisions(old, old + "\nNew unrelated note.\n")
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    items = [
        SupportWorkItem(f"w{i}", replace(memory(), id=f"memory-{i}", content=rule), (part(base, rule),), context)
        for i, rule in enumerate(rules)
    ]
    reading = reading_order(items)
    assert len(set(reading.first_part_end.values())) == len(rules)

    class OwnRuleClient(Client):
        def judge(self, prompt):
            data = payload(prompt)
            return [
                supported(work, work["prior_evidence"][0]["current_ref"]) if work["may_conclude"] else continued(work)
                for work in data["works"]
            ]

    client = OwnRuleClient(limit=100000)
    results = await RevisionWorkExecutor(client=client, model="fixture").assess_many(items)
    assert len(client.prompts) == 1 and all(result.supported for result in results.values())


@pytest.mark.asyncio
async def test_modified_legacy_part_sends_its_excerpt_with_the_current_text_it_locates():
    notes = "\n\n".join(f"Routine note {i}." for i in range(80))
    base, _ = revisions(f"{notes}\n\n{RULE}\n", "")
    # The same revision again: nothing changed, but the legacy part has no digest to prove it.
    context = RevisionAssessmentContext(projection=base, base=base, access_context_hash="scope")
    item = SupportWorkItem("w0", memory(), (part(base, RULE, exact=False),), context)
    client = Client(limit=4000)
    [result] = (await RevisionWorkExecutor(client=client, model="fixture").assess_many([item])).values()
    first = payload(client.prompts[0])
    assert first["works"][0]["prior_evidence"] == [{"role": "primary", "historical_excerpt": RULE}]
    assert any(text.strip() == RULE for _ref, text in readable(first))
    assert len(client.prompts) == 1 and result.supported and selected_texts(result) == [RULE]


@pytest.mark.asyncio
async def test_ambiguous_part_reads_every_candidate_first_and_needs_no_accounting():
    notes = "\n\n".join(f"Routine note {i}." for i in range(80))
    base, current = revisions(f"{RULE}\n\n{notes}\n", f"{RULE}\n\n{notes}\n\n{RULE}\n")
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    item = SupportWorkItem("w0", memory(), (part(base, RULE),), context)
    reading = reading_order([item])
    first_part = reading.parts[:reading.first_part_end["w0"]]
    assert [[f.presentation_text for f in p.fragments] for p in first_part] == [[RULE], [RULE]]

    class FirstRuleClient(Client):
        def judge(self, prompt):
            data = payload(prompt)
            [work] = data["works"]
            rule = [ref for ref, text in readable(data) if text.strip() == RULE]
            if work["may_conclude"]:
                # An AMBIGUOUS part has no matched ref to account for, so nothing is omitted.
                return [{**supported(work, rule[0]), "omitted_matched_refs": []}]
            return [continued(work, rule)]

    client = FirstRuleClient(limit=100000)
    [result] = (await RevisionWorkExecutor(client=client, model="fixture").assess_many([item])).values()
    assert result.supported and selected_texts(result) == [RULE]
    [request] = [payload(p) for p in client.prompts]
    assert request["works"][0]["prior_evidence"] == [{"role": "primary", "historical_excerpt": RULE}]
    assert "Routine note 79" not in client.prompts[0]


class NoCallClient(Client):
    async def evaluate_revision_work(self, prompt, **kwargs):
        raise AssertionError("this Support must be decided by the program")


@pytest.mark.asyncio
async def test_no_current_content_is_unsupported_without_a_call():
    base, current = revisions(f"{RULE}\n", f"{RULE}\n")
    # Every former member is gone from the complete target snapshot.
    unit = replace(current.source_unit_revisions[0], observation_revision_ids=())
    current = replace(current, source_unit_revisions=(unit,))
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    assert not context.full_fragments
    item = SupportWorkItem("w0", memory(), old_support(base), context)
    store = Store()
    executor = RevisionWorkExecutor(client=NoCallClient(), model="fixture", store=store, derivation_id="root")
    [result] = (await executor.assess_many([item])).values()
    assert result.supported is False and result.memory is None and executor.calls == 0
    # The empty reading order is read completely, and its program receipt says so.
    assert result.complete_read
    [receipt] = receipts(store)
    assert receipt.manifest["coverage"] == {"total": 0, "read_parts": {"w0": 0}}
    assert executor.final_work_ids == [receipt.id]


@pytest.mark.asyncio
async def test_unchanged_revision_rebinds_exact_support_without_a_model_call():
    client, store = NoCallClient(), Store()
    executor = RevisionWorkExecutor(client=client, model="fixture", store=store, derivation_id="root")
    [result] = (await executor.assess_many(work_items(f"{RULE}\n"))).values()
    assert result.supported is True and result.unresolved is None
    [primary] = result.memory.resolved_evidence_selection.parts
    assert primary.anchor.observation_revision_id == "rev-primary-v2"
    assert result.memory.support_validation["route"] == "rebind_support"
    assert result.memory.support_validation["model"] is None
    assert executor.program_rebind_count == 1 and executor.calls == 0
    # No model work happened, so there is nothing to resume and no receipt.
    assert not store.works and not executor.final_work_ids


@pytest.mark.asyncio
async def test_single_reading_group_over_capacity_is_unresolved_capacity(caplog):
    huge = "\n".join(f"- Huge item {i}: " + "detail " * 40 for i in range(40))
    other = "Hotfixes need one reviewer."
    base_text = f"{RULE}\n\n{other}\n\n{huge}\n"
    base, current = revisions(base_text, base_text + "\nNew unrelated note.\n")
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    items = [
        SupportWorkItem("rule", memory(), (part(base, RULE),), context),
        SupportWorkItem("other", replace(memory(), id="other", content=other), (part(base, other),), context),
    ]

    class ListTooLargeClient(Client):
        def request_fits(self, prompt, **kwargs):
            return "Huge item" not in prompt and super().request_fits(prompt, **kwargs)

        def judge(self, prompt):
            # The rule claim is supported once it may conclude; the other claim keeps reading.
            data = payload(prompt)
            rule = [ref for ref, text in readable(data) if text.strip() == RULE]
            return [
                supported(w, rule[0]) if w["work_id"] == "WRK-0000" and w["may_conclude"] else continued(w)
                for w in data["works"]
            ]

    client = ListTooLargeClient(limit=100000)
    executor = RevisionWorkExecutor(client=client, model="fixture")
    with caplog.at_level(logging.WARNING, logger="memforge.pipeline.revision_work"):
        results = await executor.assess_many(items)
    assert results["rule"].supported is True
    assert results["other"].supported is None and results["other"].unresolved == "capacity"
    [huge_list] = [
        p for p in reading_order(items).parts
        if p.fragments and "Huge item 0" in p.fragments[0].presentation_text
    ]
    assert "Huge item 39" in huge_list.fragments[-1].presentation_text
    [record] = [r.getMessage() for r in caplog.records if "support_unresolved_capacity" in r.getMessage()]
    assert f"source_unit_id={current.source_units[0].id}" in record
    assert "memory_id=other" in record and f"reading_group={huge_list.label} " in record
    assert "carried_witnesses=0" in record


@pytest.mark.asyncio
async def test_transient_failure_raises_after_split_to_one_item():
    error = StructuredLlmError("deadline", terminal_category="deadline_exceeded", error_code="logical_deadline_exceeded")

    class TimeoutClient(Client):
        async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
            if response_format is SupportAssessmentWireResponse and "WRK-0001" in prompt:
                self.prompts.append(prompt)
                raise error
            return await super().evaluate_revision_work(prompt, response_format=response_format, **kwargs)

    client = TimeoutClient(limit=16000)
    with pytest.raises(StructuredLlmError) as raised:
        await RevisionWorkExecutor(client=client, model="fixture").assess_many(work_items(CHANGED, 2))
    assert raised.value is error
    # The runner split the request down to the failing item alone before giving up.
    assert any("WRK-0001" in p and "WRK-0000" not in p for p in client.prompts)


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["omitted_after_correction", "selected", "never_accounted", "omits_other_refs"])
async def test_multi_part_support_accounts_for_matched_parts(variant):
    scope_old, scope_new = "Scope: US releases only.", "Scope: US releases only!"
    base, current = revisions(f"{RULE}\n\n{scope_old}\n", f"{RULE}\n\n{scope_new}\n")
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    item = SupportWorkItem(
        "w0", memory(), (part(base, RULE), part(base, scope_old, role=EvidenceRole.REQUIRED)), context,
    )

    class AccountingClient(Client):
        def judge(self, prompt):
            data = payload(prompt)
            refs = {text: ref for ref, text in readable(data)}
            [work] = data["works"]
            [matched] = [p["current_ref"] for p in work["prior_evidence"] if "current_ref" in p]
            assert matched == refs[RULE]
            if variant == "selected":
                return [supported(work, refs[RULE], [refs[scope_new]])]
            row = supported(work, refs[scope_new])
            if variant == "omits_other_refs":
                # Omitting a supplied ref that is not prior Evidence changes nothing and is accepted.
                row["omitted_matched_refs"] = [matched, refs[scope_new]]
                return [row]
            accounted = variant == "omitted_after_correction" and "<correction>" in prompt
            row["omitted_matched_refs"] = [matched] if accounted else []
            return [row]

    client = AccountingClient()
    executor = RevisionWorkExecutor(client=client, model="fixture")
    [result] = (await executor.assess_many([item])).values()
    if variant == "never_accounted":
        assert (result.supported, result.unresolved) == (None, "invalid_response")
        assert len(client.prompts) == 2 and "unaccounted" in client.prompts[-1]
        return
    assert result.supported
    if variant in {"selected", "omits_other_refs"}:
        expected = [RULE, scope_new] if variant == "selected" else [scope_new]
        assert selected_texts(result) == expected and len(client.prompts) == 1
    else:
        assert selected_texts(result) == [scope_new]
        assert len(client.prompts) == 2 and "unaccounted" in client.prompts[1]


@pytest.mark.asyncio
async def test_jira_partial_projection_missing_comment_is_unresolved_without_a_call():
    from memforge.memory.evidence import ActiveSupportEvidence
    from tests.test_source_projection_adapters import _inputs, _item, _jira_payload, project_source_item

    jira_item = _item(item_id="jira-PAY-12", extra={"issue_key": "PAY-12"})

    def jira(run_id, description, comments, *, total=None, prior=None):
        raw, normalized = _inputs(
            jira_item, _jira_payload(field_overrides={"description": description}, comments=comments, comments_total=total)
        )
        return project_source_item(
            source_id="src-j", source_type="jira", run_id=run_id, item=jira_item, raw=raw, normalized=normalized,
            prior_unit_revision=prior.source_unit_revisions[0] if prior else None,
            prior_observation_revisions={r.observation_id: r for r in prior.observation_revisions} if prior else None,
        )

    first = jira("jira-1", "Payroll approval needs two reviewers.", [{"id": "501", "body": "Decision: retain A7"}])
    # The description is returned and rewritten; comment pagination returns nothing.
    partial = jira("jira-2", "Payroll approval needs one reviewer.", [], total=1, prior=first)
    assert partial.carried_observation_revision_ids

    def support(text):
        fragment = next(
            f for f in RevisionAssessmentContext(projection=first, base=None, access_context_hash="scope").full_fragments
            if f.presentation_text == text
        )
        return (ActiveSupportEvidence(
            memory_id=text, source_id="src-j", reference_id=text, evidence_unit_id=text, role=EvidenceRole.PRIMARY,
            anchor=fragment.anchor, excerpt=text,
            raw_content_sha256=fragment.raw_content_sha256, presentation_sha256=fragment.presentation_sha256,
        ),)

    class UnsupportedClient(Client):
        def judge(self, prompt):
            data = payload(prompt)
            return [unsupported(w) if data["last"] else continued(w) for w in data["works"]]

    context = RevisionAssessmentContext(projection=partial, base=first, access_context_hash="scope")
    comment_claim = replace(memory(), id="comment", content="PAY-12 retains A7.")
    description_claim = replace(memory(), id="description", content="Payroll approval needs two reviewers.")
    items = [
        SupportWorkItem("comment", comment_claim, support("Decision: retain A7"), context),
        SupportWorkItem("description", description_claim, support("Payroll approval needs two reviewers."), context),
    ]
    client = UnsupportedClient()
    results = await RevisionWorkExecutor(client=client, model="fixture").assess_many(items)

    assert results["comment"].supported is None and results["comment"].memory is None
    assert results["comment"].unresolved == "partial_coverage"
    # The description's coverage is authoritative, so reading the whole order can conclude UNSUPPORTED.
    assert results["description"].supported is False and results["description"].unresolved is None
    assert client.prompts and not any(comment_claim.content in prompt for prompt in client.prompts)
