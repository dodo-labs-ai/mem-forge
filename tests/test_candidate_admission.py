"""Candidate admission: complete Evidence support and same-round deduplication."""

from dataclasses import replace

import pytest

from memforge.llm.structured import CandidateAdmissionResponse, StructuredLlmError
from memforge.memory.candidate_admission import CandidateAdmissionError, admit_candidates
from memforge.models import RawMemory
from memforge.pipeline.complete_support import COMPLETE_SUPPORT_DEFINITION
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from tests.llm_fixture import (
    FIXTURE_CONTEXT_WINDOW,
    FIXTURE_EXTRACTION_OUTPUT_TOKENS,
    admission_payload,
    fixture_budget,
)
from tests.test_derivation_work import prepare_database
from tests.test_revision_assessment import revisions

MOST_SPECIFIC = "US releases require two reviewers from distinct teams before approval."
SAME_KNOWLEDGE = "US releases need two reviewers from distinct teams."
INDEPENDENT = "Two reviewers are required."


def admitted(candidate, _round):
    return {"candidate_id": candidate["id"], "verdict": "ADMITTED"}


class AdmissionClient:
    """Answers each requested Candidate with ``judge(candidate, round_claims)``."""

    input_policy_identity = "fixture-input-policy"

    def __init__(self, judge=admitted, *, fits=lambda payload: True, fail_at=None):
        self.judge = judge
        self.fits = fits
        self.fail_at = fail_at
        self.calls = 0
        self.requests: list[dict] = []
        self.prompts: list[str] = []

    def request_budget(self, model=None):
        return fixture_budget(
            input_tokens=FIXTURE_CONTEXT_WINDOW, output_tokens=FIXTURE_EXTRACTION_OUTPUT_TOKENS, correction_reserve=0,
        )

    def request_fits(self, prompt, **kwargs):
        return self.fits(admission_payload(prompt))

    def input_policy_identity_for(self, model=None):
        return f"{self.input_policy_identity}:{model}"

    async def admit_candidates(self, prompt, **kwargs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise StructuredLlmError("fixture deadline", terminal_category="deadline_exceeded",
                                     error_code="logical_deadline_exceeded")
        payload = admission_payload(prompt)
        self.prompts.append(prompt)
        self.requests.append(payload)
        round_claims = {row["id"]: row["claim"] for row in payload["round_claims"]}
        return CandidateAdmissionResponse.model_validate({"decisions": [
            self.judge(candidate, round_claims) for candidate in payload["candidates"]
        ]})


def candidate(content: str, *, required: bool = True) -> RawMemory:
    _, target = revisions("Two reviewers required.\n", "Two reviewers from distinct teams required.\n")
    context = RevisionAssessmentContext(projection=target, base=None, access_context_hash="scope")
    catalog = context.catalog(context.full_fragments)
    primary = next(f.reference for f in catalog.fragments if "distinct" in f.presentation_text)
    required_refs = [next(f.reference for f in catalog.fragments if "Country" in f.presentation_text)] if required else []
    return RawMemory(
        content=content,
        memory_type="fact",
        resolved_evidence_selection=catalog.resolve_selection(primary_ref=primary, required_refs=required_refs),
    )


def ids_by_claim(client: AdmissionClient) -> dict[str, str]:
    return {row["claim"]: row["id"] for request in client.requests for row in request["round_claims"]}


@pytest.mark.asyncio
async def test_a_single_candidate_is_judged_with_exactly_its_selected_evidence():
    client = AdmissionClient()
    only = candidate(MOST_SPECIFIC)

    result = await admit_candidates([only], client=client, model="fixture")

    assert result.admitted == (only,) and result.rejected == () and result.merged_count == 0
    assert result.llm_calls == 1
    [request] = client.requests
    [row] = request["candidates"]
    assert row["claim"] == MOST_SPECIFIC
    assert [request["evidence_catalog"][ref]["role"] for ref in row["evidence_refs"]] == ["primary", "required"]
    assert [request["evidence_catalog"][ref]["excerpt"] for ref in row["evidence_refs"]] == [
        "Two reviewers from distinct teams required.", "Country: US.",
    ]
    assert request["round_claims"] == [{"id": row["id"], "claim": MOST_SPECIFIC}]
    assert COMPLETE_SUPPORT_DEFINITION in client.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_reason", ["evidence_incomplete", "low_value"])
async def test_rejected_candidate_is_not_admitted_and_keeps_its_reason(reject_reason):
    def judge(row, _round):
        if row["claim"] == SAME_KNOWLEDGE:
            return {"candidate_id": row["id"], "verdict": "REJECTED", "reject_reason": reject_reason}
        return admitted(row, _round)

    kept, dropped = candidate(MOST_SPECIFIC), candidate(SAME_KNOWLEDGE, required=False)
    result = await admit_candidates([kept, dropped], client=AdmissionClient(judge), model="fixture")

    assert result.admitted == (kept,)
    assert [(rejection.candidate, rejection.reject_reason) for rejection in result.rejected] == [(dropped, reject_reason)]
    assert result.merged_count == 0


@pytest.mark.asyncio
async def test_duplicates_judged_in_different_requests_merge_into_the_most_specific_candidate():
    # One Candidate and at most two round claims per request: each Candidate reads
    # the round in two chunks and never shares a request with another Candidate.
    def fits(payload):
        return len(payload["candidates"]) == 1 and len(payload["round_claims"]) <= 2

    def judge(row, round_claims):
        duplicate = [ref for ref, claim in round_claims.items() if claim == MOST_SPECIFIC]
        if row["claim"] == SAME_KNOWLEDGE and duplicate:
            return {"candidate_id": row["id"], "verdict": "ADMITTED", "duplicate_of": duplicate}
        return admitted(row, round_claims)

    client = AdmissionClient(judge, fits=fits)
    specific, same, independent = candidate(MOST_SPECIFIC), candidate(SAME_KNOWLEDGE), candidate(INDEPENDENT)

    result = await admit_candidates([same, independent, specific], client=client, model="fixture")

    assert result.admitted == (independent, specific)
    assert result.merged_count == 1 and result.rejected == ()
    assert all(len(request["candidates"]) == 1 for request in client.requests)
    assert len(client.requests) == 6


@pytest.mark.asyncio
async def test_a_rejection_in_any_context_chunk_rejects_the_candidate():
    def fits(payload):
        return len(payload["candidates"]) == 1 and len(payload["round_claims"]) <= 1

    def judge(row, round_claims):
        if row["claim"] == SAME_KNOWLEDGE and INDEPENDENT in round_claims.values():
            return {"candidate_id": row["id"], "verdict": "REJECTED", "reject_reason": "evidence_incomplete"}
        return admitted(row, round_claims)

    same, independent = candidate(SAME_KNOWLEDGE), candidate(INDEPENDENT)
    result = await admit_candidates([same, independent], client=AdmissionClient(judge, fits=fits), model="fixture")

    assert result.admitted == (independent,)
    assert [rejection.candidate for rejection in result.rejected] == [same]


@pytest.mark.asyncio
async def test_a_rejected_candidate_neither_absorbs_nor_links_duplicates():
    def judge(row, round_claims):
        ids = {claim: ref for ref, claim in round_claims.items()}
        if row["claim"] == SAME_KNOWLEDGE:
            return {"candidate_id": row["id"], "verdict": "REJECTED", "reject_reason": "low_value",
                    "duplicate_of": [ids[MOST_SPECIFIC], ids[INDEPENDENT]]}
        return admitted(row, round_claims)

    specific, same, independent = candidate(MOST_SPECIFIC), candidate(SAME_KNOWLEDGE), candidate(INDEPENDENT)
    result = await admit_candidates([specific, same, independent], client=AdmissionClient(judge), model="fixture")

    assert result.admitted == (specific, independent)
    assert result.merged_count == 0


def repeated(content: str, *, required: bool = True) -> RawMemory:
    return replace(candidate(content, required=required), content=f"  {content.replace(' ', '   ')} ")


@pytest.mark.asyncio
async def test_identical_claims_are_each_judged_and_merge_once_admitted():
    client = AdmissionClient()
    first, second = candidate(MOST_SPECIFIC), repeated(MOST_SPECIFIC)

    result = await admit_candidates([first, second], client=client, model="fixture")

    assert result.admitted == (first,) and result.merged_count == 1 and result.rejected == ()
    [request] = client.requests
    assert len(request["candidates"]) == 2


@pytest.mark.asyncio
async def test_an_identical_claim_with_complete_evidence_survives_a_rejected_copy():
    def judge(row, _round):
        if row["evidence_refs"] and len(row["evidence_refs"]) == 1:
            return {"candidate_id": row["id"], "verdict": "REJECTED", "reject_reason": "evidence_incomplete"}
        return admitted(row, _round)

    incomplete, complete = candidate(MOST_SPECIFIC, required=False), repeated(MOST_SPECIFIC)

    result = await admit_candidates([incomplete, complete], client=AdmissionClient(judge), model="fixture")

    assert result.admitted == (complete,)
    assert [rejection.candidate for rejection in result.rejected] == [incomplete]
    assert result.merged_count == 0


@pytest.mark.asyncio
async def test_identical_claims_of_different_validity_are_not_merged_by_the_program():
    first = candidate(MOST_SPECIFIC)
    later = replace(repeated(MOST_SPECIFIC), valid_from="2027-01-01")

    result = await admit_candidates([first, later], client=AdmissionClient(), model="fixture")

    assert result.admitted == (first, later) and result.merged_count == 0


@pytest.mark.asyncio
async def test_no_candidates_need_no_model_call():
    client = AdmissionClient()
    result = await admit_candidates([], client=client, model="fixture")
    assert result.admitted == () and client.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "judge",
    [
        pytest.param(lambda row, _round: {"candidate_id": "CND-0099", "verdict": "ADMITTED"}, id="unknown-id"),
        pytest.param(lambda row, _round: {"candidate_id": row["id"], "verdict": "ADMITTED",
                                          "duplicate_of": ["CND-0099"]}, id="duplicate-outside-round"),
    ],
)
async def test_a_candidate_whose_admission_stays_invalid_is_rejected_for_this_round(judge):
    unjudged = candidate(MOST_SPECIFIC)
    judged = candidate(INDEPENDENT)

    def answer(row, round_claims):
        return judge(row, round_claims) if round_claims.get(row["id"]) == MOST_SPECIFIC else admitted(row, round_claims)

    client = AdmissionClient(answer)
    result = await admit_candidates([unjudged, judged], client=client, model="fixture")

    assert result.admitted == (judged,)
    assert [(rejection.candidate, rejection.reject_reason) for rejection in result.rejected] == [
        (unjudged, "invalid_response"),
    ]
    # Both Candidates with the correction, then each alone: the invalid one with its correction.
    assert client.calls == 5


@pytest.mark.asyncio
async def test_a_candidate_that_alone_exceeds_capacity_is_rejected_for_this_round():
    oversized = candidate(MOST_SPECIFIC)
    judged = candidate(INDEPENDENT)

    def fits(payload):
        return all(row["claim"] != MOST_SPECIFIC for row in payload["candidates"])

    result = await admit_candidates([oversized, judged], client=AdmissionClient(fits=fits), model="fixture")

    assert result.admitted == (judged,)
    assert [(rejection.candidate, rejection.reject_reason) for rejection in result.rejected] == [
        (oversized, "capacity_exceeded"),
    ]


@pytest.mark.asyncio
async def test_a_transient_admission_failure_raises():
    with pytest.raises(StructuredLlmError) as raised:
        await admit_candidates([candidate(MOST_SPECIFIC)], client=AdmissionClient(fail_at=1), model="fixture")
    assert raised.value.terminal_category == "deadline_exceeded"


@pytest.mark.asyncio
async def test_a_candidate_without_selected_evidence_is_an_execution_failure():
    with pytest.raises(CandidateAdmissionError) as raised:
        await admit_candidates([RawMemory(content=MOST_SPECIFIC, memory_type="fact")],
                               client=AdmissionClient(), model="fixture")
    assert (raised.value.reason_code, raised.value.terminal_category) == ("candidate_evidence_missing", None)


@pytest.mark.asyncio
async def test_a_retried_sync_reuses_completed_admission_requests(tmp_path):
    from memforge.storage.database import Database

    path = tmp_path / "admission.db"
    db, root = await prepare_database(path)
    candidates = [candidate(MOST_SPECIFIC), candidate(SAME_KNOWLEDGE), candidate(INDEPENDENT)]
    kwargs = dict(model="fixture", derivation_id=root.id, operation_input_hash="a" * 64)

    def one_candidate_per_request(payload):
        return len(payload["candidates"]) == 1

    try:
        with pytest.raises(StructuredLlmError, match="fixture deadline"):
            await admit_candidates(
                candidates, client=AdmissionClient(fits=one_candidate_per_request, fail_at=3), store=db, **kwargs,
            )
    finally:
        await db.close()
    db = Database(str(path))
    await db.connect()
    try:
        retry = AdmissionClient(fits=one_candidate_per_request)
        result = await admit_candidates(candidates, client=retry, store=db, **kwargs)
        assert retry.calls == 1
        assert result.admitted == tuple(candidates)
        assert len(result.work_ids) == 3
        async with db.db.execute(
            "SELECT COUNT(*) AS n FROM source_derivation_work WHERE derivation_id = ? "
            "AND json_extract(payload_json, '$.kind') = 'candidate_admission' "
            "AND json_extract(payload_json, '$.status') = 'completed'",
            (root.id,),
        ) as cursor:
            assert (await cursor.fetchone())["n"] == 3
    finally:
        await db.close()
