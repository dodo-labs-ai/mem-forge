"""SupportRelationCoordinator end to end: concurrent lines, re-checks and pending coordinator Reviews."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memforge.llm.structured import ClaimRevisionWireResponse, StructuredLlmError
from memforge.memory.coordinator_review import coordinator_review_id
from memforge.memory.lifecycle_plan import LifecycleReviewStatus
from memforge.memory.lifecycle_review import (
    build_lifecycle_review_approval_plan,
    build_lifecycle_review_rejection_plan,
)
from memforge.models import CoordinatorProposal, RawMemory
from memforge.storage.database import Database
from tests.coordination_fixture import (
    AUDIT,
    DOC_ID,
    JIRA_DOCUMENT,
    ONE,
    RESTATED,
    RETENTION,
    SOURCE_ID,
    TWO,
    ScriptedClient,
    coordination_engine,
    lifecycle_reviews,
    partial_jira_revision,
    seed_jira_issue,
    seeded_page,
    support_texts,
)
from tests.revision_client_fixture import catalog_payload
from tests.test_projected_lifecycle_integration import db as db, _selected
from tests.unit_support_fixture import active_support_evidence

# Bounds how long one line waits to see the other start; a serial implementation never does.
_CONCURRENCY_TIMEOUT_S = 5


@pytest.mark.asyncio
async def test_a_contradicted_supported_claim_keeps_one_review_across_revisions(db: Database) -> None:
    page, memory = await seeded_page(db, TWO, RETENTION)
    client = ScriptedClient(relations={(ONE, TWO): "contradicts"})

    # The source now states both claims: the old one is rebound and the conflict goes to a Review.
    stats = await page.commit(client, page.next(TWO, ONE, RETENTION), ONE)
    [review] = await lifecycle_reviews(db, memory.id)
    assert review.id == coordinator_review_id(page.unit_id, memory.id, CoordinatorProposal.SUPERSEDE, ONE)
    assert review.status is LifecycleReviewStatus.PENDING
    assert review.staged_evidence["proposal"] == CoordinatorProposal.SUPERSEDE.value
    assert review.staged_evidence["stale_guard"]["support_set_hash"] == await db.get_memory_support_set_hash(memory.id)
    assert stats["coordinator_review_count"] == 1 and stats["pending_review"] == 1
    assert [item.id for item in await db.list_memories()] == [memory.id]
    rebound = await active_support_evidence(db, memory.id, source_id=SOURCE_ID)
    assert {part.validation_unit_revision_id for part in rebound} == {page.current.source_unit_revisions[0].id}

    # An unrelated edit does not extract the Candidate again; its exact Evidence carries the conflict.
    stats = await page.commit(client, page.next(TWO, ONE, RETENTION, AUDIT))
    [carried] = await lifecycle_reviews(db, memory.id)
    assert (carried.id, carried.status) == (review.id, LifecycleReviewStatus.PENDING)
    assert carried.lifecycle_plan_id != review.lifecycle_plan_id
    assert stats["coordinator_carried_conflict_count"] == 1 and stats["coordinator_recheck_count"] == 0
    assert [item.id for item in await db.list_memories()] == [memory.id]

    # The contradicting statement is gone, so the conflict is gone.
    await page.commit(client, page.next(TWO, RETENTION, AUDIT))
    [closed] = await lifecycle_reviews(db, memory.id)
    assert (closed.id, closed.status) == (review.id, LifecycleReviewStatus.STALE)

    # It comes back and reopens the same Review, which a reviewer then approves.
    await page.commit(client, page.next(TWO, ONE, RETENTION, AUDIT), ONE)
    [reopened] = await lifecycle_reviews(db, memory.id)
    assert (reopened.id, reopened.status, reopened.resolved_at) == (review.id, LifecycleReviewStatus.PENDING, None)
    payload = await db.get_lifecycle_plan_payload(reopened.lifecycle_plan_id)
    await db.apply_lifecycle_plan(build_lifecycle_review_approval_plan(reopened, payload))

    superseded = await db.get_memory(memory.id)
    assert superseded is not None and superseded.status == "superseded"
    assert [item.content for item in await db.list_memories() if item.status == "active"] == [ONE]


@pytest.mark.asyncio
async def test_a_rejected_conflict_is_not_raised_again(db: Database) -> None:
    page, memory = await seeded_page(db, TWO, RETENTION)
    client = ScriptedClient(relations={(ONE, TWO): "contradicts"})
    await page.commit(client, page.next(TWO, ONE, RETENTION), ONE)
    [review] = await lifecycle_reviews(db, memory.id)
    await db.resolve_lifecycle_review(review.id, LifecycleReviewStatus.REJECTED, review_note="Two reviewers stand")

    stats = await page.commit(client, page.next(TWO, ONE + " ", RETENTION), ONE)

    [kept] = await lifecycle_reviews(db, memory.id)
    assert (kept.id, kept.status) == (review.id, LifecycleReviewStatus.REJECTED)
    assert stats["pending_review"] == 0 and stats["added"] == 0
    assert [(item.id, item.status) for item in await db.list_memories()] == [(memory.id, "active")]


@pytest.mark.asyncio
async def test_an_unaffected_claim_contradicted_by_the_source_is_read_once_in_the_normal_order(db: Database) -> None:
    page, memory = await seeded_page(db, TWO, RETENTION)
    client = ScriptedClient(relations={(ONE, TWO): "contradicts"}, impact="unaffected")

    stats = await page.commit(client, page.next(TWO, RETENTION, ONE), ONE)

    assert stats["support_revalidation_change_impact_unaffected_count"] == 1
    assert stats["coordinator_recheck_count"] == 1
    # The re-check is an ordinary ordered read that may reach every group of the revision.
    [recheck] = client.support_requests
    assert TWO in client.supplied_texts(recheck)
    [review] = await lifecycle_reviews(db, memory.id)
    assert review.staged_evidence["proposal"] == CoordinatorProposal.SUPERSEDE.value


@pytest.mark.asyncio
@pytest.mark.parametrize("recheck_supported", [True, False])
async def test_an_unsupported_claim_with_an_equivalent_is_rechecked_against_that_evidence(
    db: Database, recheck_supported: bool,
) -> None:
    page, memory = await seeded_page(db, TWO, RETENTION)
    client = ScriptedClient(
        relations={(RESTATED, TWO): "equivalent"},
        support_text={TWO: RESTATED},
        verdicts={TWO: [False, recheck_supported]},
    )

    stats = await page.commit(client, page.next(RESTATED, RETENTION), RESTATED)

    *first_read, recheck = client.support_requests
    assert any(RETENTION in client.supplied_texts(request) for request in first_read)
    # The re-check reads only the ReadingGroup that holds the Candidate's Evidence, with the page title.
    assert RESTATED in client.supplied_texts(recheck) and RETENTION not in client.supplied_texts(recheck)
    assert stats["coordinator_recheck_count"] == 1
    assert [item.id for item in await db.list_memories()] == [memory.id]
    if recheck_supported:
        assert await lifecycle_reviews(db, memory.id) == []
        assert await support_texts(db, memory.id) == {RESTATED}
        return
    [review] = await lifecycle_reviews(db, memory.id)
    assert review.staged_evidence["proposal"] == CoordinatorProposal.REBIND.value
    assert review.staged_evidence["candidate"]["content"] == RESTATED
    assert await support_texts(db, memory.id) == {TWO}
    payload = await db.get_lifecycle_plan_payload(review.lifecycle_plan_id)
    await db.apply_lifecycle_plan(build_lifecycle_review_approval_plan(review, payload))
    assert (await db.get_memory(memory.id)).status == "active"
    assert await support_texts(db, memory.id) == {RESTATED}


@pytest.mark.asyncio
async def test_a_failed_recheck_leaves_the_revision_uncommitted_and_a_retry_keeps_the_candidate(db: Database) -> None:
    page, memory = await seeded_page(db, TWO, RETENTION)

    class FailingRecheck(ScriptedClient):
        async def assess_support(self, prompt, **kwargs):
            if not self.verdicts[TWO]:
                raise StructuredLlmError("provider unavailable", terminal_category="provider_error")
            return await super().assess_support(prompt, **kwargs)

    committed = page.current
    revision = page.next(RESTATED, RETENTION)
    relations = {(RESTATED, TWO): "equivalent"}
    with pytest.raises(StructuredLlmError):
        await page.commit(FailingRecheck(relations=relations, verdicts={TWO: [False]}), revision, RESTATED)
    current = await db.get_current_source_unit_projection(committed.source_units[0].id)
    assert current.source_unit_revisions[0].id == committed.source_unit_revisions[0].id
    assert await lifecycle_reviews(db, memory.id) == []

    await page.commit(
        ScriptedClient(relations=relations, support_text={TWO: RESTATED}, verdicts={TWO: [False, True]}),
        revision, RESTATED,
    )

    assert [item.id for item in await db.list_memories()] == [memory.id]
    assert await support_texts(db, memory.id) == {RESTATED}


@pytest.mark.asyncio
async def test_rejecting_a_conflict_with_an_equivalent_binds_the_claim_to_that_equivalent(db: Database) -> None:
    page, memory = await seeded_page(db, TWO, RETENTION)
    client = ScriptedClient(
        relations={(RESTATED, TWO): "equivalent", (ONE, TWO): "contradicts"}, verdicts={TWO: [False]},
    )

    stats = await page.commit(client, page.next(RESTATED, ONE, RETENTION), RESTATED, ONE)

    assert stats["coordinator_recheck_count"] == 0
    [review] = await lifecycle_reviews(db, memory.id)
    assert review.staged_evidence["candidate"]["content"] == ONE
    assert review.staged_evidence["rejection_candidate"]["content"] == RESTATED
    payload = await db.get_lifecycle_plan_payload(review.lifecycle_plan_id)
    rejection = build_lifecycle_review_rejection_plan(review, payload, review_note="Two reviewers stand")
    assert rejection is not None
    await db.apply_lifecycle_plan(rejection)

    [rejected] = await lifecycle_reviews(db, memory.id)
    assert rejected.status is LifecycleReviewStatus.REJECTED
    assert (await db.get_memory(memory.id)).status == "active"
    assert await support_texts(db, memory.id) == {RESTATED}


class _OmitsRow(ScriptedClient):
    """Never returns the Relation row of one Candidate, even after the correction."""

    def __init__(self, omitted: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.omitted = omitted

    async def assess_claim_revisions(self, prompt, **kwargs):
        response = await super().assess_claim_revisions(prompt, **kwargs)
        texts = {claim["id"]: claim["text"] for claim in catalog_payload(prompt)["new_claims"]}
        return ClaimRevisionWireResponse(results=[
            row for row in response.results if texts[row.candidate_id] != self.omitted
        ])


@pytest.mark.asyncio
async def test_a_candidate_whose_relation_stays_invalid_is_unresolved_and_the_revision_commits(
    db: Database, caplog,
) -> None:
    page, memory = await seeded_page(db, TWO, RETENTION)

    revision = page.next(TWO, RETENTION, AUDIT)
    with caplog.at_level("WARNING", logger="memforge.memory.engine"):
        stats = await page.commit(_OmitsRow(AUDIT), revision, RETENTION, AUDIT)

    current = await db.get_current_source_unit_projection(page.unit_id)
    assert current.source_unit_revisions[0].id == revision.source_unit_revisions[0].id
    # The old claim is kept and the independent Candidate added; the unjudged one is consumed without ADD or Review.
    assert sorted(item.content for item in await db.list_memories()) == sorted([TWO, RETENTION])
    assert await support_texts(db, memory.id) == {TWO}
    assert await lifecycle_reviews(db, memory.id) == []
    assert stats["added"] == 1 and stats["relation_unjudged_candidate_count"] == 1
    assert stats["coordinator_unresolved_candidate_count"] == 1
    [record] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("relation_candidate_unresolved")]
    assert f"source_unit_id={page.unit_id}" in record and "reason=invalid_response" in record


@pytest.mark.asyncio
async def test_an_unjudged_restatement_keeps_the_claim_it_may_restate(db: Database) -> None:
    page, memory = await seeded_page(db, TWO, RETENTION)
    old_support = await support_texts(db, memory.id)

    # The revision rewords the claim: Support reads the whole revision without finding TWO,
    # and Relation cannot judge the restating Candidate.
    revision = page.next(RESTATED, RETENTION)
    stats = await page.commit(_OmitsRow(RESTATED), revision, RESTATED)

    current = await db.get_current_source_unit_projection(page.unit_id)
    assert current.source_unit_revisions[0].id == revision.source_unit_revisions[0].id
    kept = await db.get_memory(memory.id)
    assert kept.status == "active" and await support_texts(db, memory.id) == old_support
    assert [item.id for item in await db.list_memories()] == [memory.id]
    assert stats["relation_unjudged_candidate_count"] == 1
    assert stats["destructive_validation_kept_relation_incomplete_count"] == 1
    assert stats["deleted"] == 0 and stats["added"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        StructuredLlmError("bad request", terminal_category="request_error", error_code="BadRequestError"),
        KeyError("results"),
    ],
    ids=["provider-400", "code-bug"],
)
async def test_a_relation_error_without_a_response_to_validate_leaves_the_revision_uncommitted(
    db: Database, error: Exception,
) -> None:
    page, memory = await seeded_page(db, TWO, RETENTION)
    committed = page.current

    class FailingRelation(ScriptedClient):
        def __init__(self) -> None:
            super().__init__()
            self.relation_calls = 0

        async def assess_claim_revisions(self, prompt, **kwargs):
            self.relation_calls += 1
            raise error

    client = FailingRelation()
    with pytest.raises(Exception):
        await page.commit(client, page.next(TWO, RETENTION, AUDIT, ONE), AUDIT, ONE)

    current = await db.get_current_source_unit_projection(committed.source_units[0].id)
    assert current.source_unit_revisions[0].id == committed.source_unit_revisions[0].id
    # Nothing was isolated item by item: the one request failed and the revision waits for the next sync.
    assert client.relation_calls == 1
    assert [item.id for item in await db.list_memories()] == [memory.id]


@pytest.mark.asyncio
async def test_support_and_relation_run_concurrently(db: Database) -> None:
    page, memory = await seeded_page(db, TWO, RETENTION)
    relation_started, support_started = asyncio.Event(), asyncio.Event()

    class Rendezvous(ScriptedClient):
        async def assess_claim_revisions(self, prompt, **kwargs):
            relation_started.set()
            await asyncio.wait_for(support_started.wait(), _CONCURRENCY_TIMEOUT_S)
            return await super().assess_claim_revisions(prompt, **kwargs)

        async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
            support_started.set()
            await asyncio.wait_for(relation_started.wait(), _CONCURRENCY_TIMEOUT_S)
            return await super().evaluate_revision_work(prompt, response_format=response_format, **kwargs)

    stats = await page.commit(Rendezvous(relations={(RESTATED, TWO): "equivalent"}), page.next(TWO, RESTATED), RESTATED)

    assert relation_started.is_set() and support_started.is_set()
    assert stats["added"] == 0 and [item.id for item in await db.list_memories()] == [memory.id]



@pytest.mark.asyncio
async def test_a_failure_raised_on_one_line_cancels_the_other_and_propagates_unchanged() -> None:
    from memforge.memory.engine import _run_concurrently

    started, cancelled = asyncio.Event(), asyncio.Event()

    async def waiting_line() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def failing_line() -> None:
        await started.wait()
        raise StructuredLlmError("provider unavailable", terminal_category="provider_error")

    with pytest.raises(StructuredLlmError, match="provider unavailable"):
        await asyncio.wait_for(_run_concurrently(waiting_line(), failing_line()), _CONCURRENCY_TIMEOUT_S)
    assert cancelled.is_set()

async def _jira_incumbent(db: Database):
    """A claim whose Evidence is a comment the next, partial, page of comments leaves out."""
    first = await seed_jira_issue(
        db, run_id="projection-jira-coordinator-1", description="Initial issue description.",
        claims=[RawMemory(content="A7 is retained.", memory_type="decision", evidence_quote="Decision: retain A7")],
    )
    [memory] = await db.list_memories()
    second = partial_jira_revision(
        first, run_id="projection-jira-coordinator-2", description="A7 stays for regular payroll.",
    )
    return memory, first, second


async def _commit_jira(db: Database, client, base, projection, claim: str):
    return await coordination_engine(db, client).prepare_and_commit_projected_lifecycle(
        projection=projection, doc_id=DOC_ID,
        raw_memories=_selected(projection, [RawMemory(
            content=claim, memory_type="decision", evidence_quote="A7 stays for regular payroll.",
        )], base=base),
        doc_type="ticket", project_key="ENG", repo_identifier=None, document_content=JIRA_DOCUMENT,
        update_mode="diff_guided", changed_hunks=None, update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_a_partial_coverage_equivalent_rebinds_the_claim_after_one_recheck(db: Database) -> None:
    memory, first, second = await _jira_incumbent(db)
    claim = "A7 stays for regular payroll."
    client = ScriptedClient(
        relations={(claim, memory.content): "equivalent"}, support_text={memory.content: claim},
    )

    stats = await _commit_jira(db, client, first, second, claim)

    assert stats["support_revalidation_unresolved_partial_coverage_count"] == 1
    assert stats["coordinator_recheck_count"] == 1 and stats["added"] == 0
    [recheck] = client.support_requests
    assert client.supplied_texts(recheck) == [claim]
    # The Support that had an UNKNOWN part is replaced by the returned description.
    assert await support_texts(db, memory.id) == {claim}
    assert await lifecycle_reviews(db, memory.id) == []


@pytest.mark.asyncio
async def test_a_partial_coverage_contradiction_is_reviewed_and_never_superseded(db: Database) -> None:
    memory, first, second = await _jira_incumbent(db)
    claim = "A7 is dropped for regular payroll."
    client = ScriptedClient(relations={(claim, memory.content): "contradicts"})
    before = await active_support_evidence(db, memory.id, source_id=SOURCE_ID)

    stats = await _commit_jira(db, client, first, second, claim)

    assert stats["superseded"] == 0 and stats["coordinator_recheck_count"] == 0
    assert client.support_requests == []
    [review] = await lifecycle_reviews(db, memory.id)
    assert review.staged_evidence["proposal"] == CoordinatorProposal.SUPERSEDE.value
    assert (await db.get_memory(memory.id)).status == "active"
    assert await active_support_evidence(db, memory.id, source_id=SOURCE_ID) == before


class _DeliveredStore:
    """Stands in for the vector store: every lifecycle Plan's vector work is delivered."""

    async def attempt_lifecycle_vector_delivery(self, lifecycle_plan_id=None, *, source_id=None):
        return SimpleNamespace(pending=False, error_types=())


async def _decide(client, review_id: str, decision: str, note: str | None = None):
    detail = client.get(f"/api/v1/memory-reviews/{review_id}")
    assert detail.status_code == 200, detail.text
    body = {"expected_fingerprint": detail.json()["decision_fingerprint"], **({"note": note} if note else {})}
    return client.post(f"/api/v1/memory-reviews/{review_id}/{decision}", json=body)


@pytest.mark.asyncio
async def test_a_coordinator_review_whose_guard_moved_waits_for_the_next_revision(
    db: Database, tmp_path, monkeypatch,
) -> None:
    from fastapi.testclient import TestClient

    from memforge.config import AppConfig
    from memforge.server.admin_api import create_admin_app

    async def delivered_store(*args, **kwargs):
        return _DeliveredStore()

    monkeypatch.setattr("memforge.server.admin_api._build_memory_store", delivered_store)
    none = "No reviewer approves payroll."
    page, memory = await seeded_page(db, TWO, RETENTION)
    client = ScriptedClient(
        relations={(RESTATED, TWO): "equivalent", (ONE, TWO): "contradicts", (none, TWO): "contradicts"},
        verdicts={TWO: [False]},
    )
    await page.commit(client, page.next(RESTATED, ONE, none, RETENTION), RESTATED, ONE, none)
    reviews = {review.staged_evidence["candidate"]["content"]: review for review in await lifecycle_reviews(db, memory.id)}
    assert set(reviews) == {ONE, none}
    assert {review.staged_evidence["rejection_candidate"]["content"] for review in reviews.values()} == {RESTATED}

    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    app = create_admin_app(db=db, config=config, runtime_provider=SimpleNamespace())
    with TestClient(app) as http:
        # Keeping the current state rebinds the old Memory to its restatement.
        rejected = await _decide(http, reviews[ONE].id, "reject", note="Two reviewers stand")
        assert rejected.status_code == 200, rejected.text
        assert await support_texts(db, memory.id) == {RESTATED}
        # That rebind moved the other Review's guard: it is refused and stays pending.
        refused = await _decide(http, reviews[none].id, "approve")
        assert refused.status_code == 409
        assert "raises this conflict again" in refused.json()["detail"]
        held = await db.get_lifecycle_review(reviews[none].id)
        assert held.status is LifecycleReviewStatus.PENDING

    # The next revision carries the conflict against the rebound Support and refreshes its guard.
    await page.commit(client, page.next(RESTATED, ONE, none, RETENTION, AUDIT))
    [rejected_review, refreshed] = sorted(
        await lifecycle_reviews(db, memory.id), key=lambda review: review.status is LifecycleReviewStatus.PENDING,
    )
    assert (rejected_review.id, rejected_review.status) == (reviews[ONE].id, LifecycleReviewStatus.REJECTED)
    assert (refreshed.id, refreshed.status) == (reviews[none].id, LifecycleReviewStatus.PENDING)
    with TestClient(app) as http:
        approved = await _decide(http, refreshed.id, "approve")
        assert approved.status_code == 200, approved.text
    assert (await db.get_memory(memory.id)).status == "superseded"
    assert [item.content for item in await db.list_memories() if item.status == "active"] == [none]
