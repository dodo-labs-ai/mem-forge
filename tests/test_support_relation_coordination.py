"""SupportRelationCoordinator end to end: concurrent lines, re-checks and pending coordinator Reviews."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from memforge.llm.structured import (
    MemoryRelationDecision,
    MemoryRelationResponse,
    StructuredLlmError,
    SupportAssessmentWireResponse,
)
from memforge.memory.coordinator_review import coordinator_review_id
from memforge.memory.engine import MemoryEngine
from memforge.memory.lifecycle_plan import LifecycleReviewStatus
from memforge.memory.lifecycle_review import (
    build_lifecycle_review_approval_plan,
    build_lifecycle_review_rejection_plan,
)
from memforge.models import CoordinatorProposal, RawMemory
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database
from tests.revision_client_fixture import RevisionClientFixture
from tests.test_projected_lifecycle_integration import (
    db as db,
    _OutboxDrainer,
    _candidate_retriever,
    _jira_projection,
    _projection,
    _selected,
    _set_fixture_source_type,
)
from tests.unit_support_fixture import active_support_evidence

TWO = "Two reviewers approve payroll."
ONE = "One reviewer approves payroll."
RESTATED = "Payroll needs two approvers."
RETENTION = "Retention is seven years."
AUDIT = "Audit logs are kept."
# Bounds how long one line waits to see the other start; a serial implementation never does.
_CONCURRENCY_TIMEOUT_S = 5


class _ScriptedClient(RevisionClientFixture):
    """Relation labels by (Candidate, old claim); Support by the Evidence text a request supplies.

    A claim is supported when a supplied Primary Fragment states ``support_text`` for it,
    unless ``verdicts`` scripts the judgment each successive read of that claim ends with.
    """

    def __init__(self, *, relations=None, support_text=None, verdicts=None, impact="affected") -> None:
        self.relations = dict(relations or {})
        self.support_text = dict(support_text or {})
        self.verdicts = {claim: list(values) for claim, values in (verdicts or {}).items()}
        self.impact = impact
        self.support_requests: list[dict] = []

    async def classify_memory_relations(self, prompt: str, **kwargs):
        groups = json.loads(prompt.split("<memory_pair_groups>\n", 1)[1].split("\n</memory_pair_groups>", 1)[0])
        decisions = []
        for group in groups:
            for item in group["candidates"]:
                label = self.relations.get((group["challenger"]["content"], item["content"]), "unrelated")
                decisions.append(MemoryRelationDecision(
                    pair_index=item["pair_index"], classification=label, direction="symmetric",
                    same_subject_and_scope=label == "contradicts",
                    incompatible_assertions="the reviewer counts differ" if label == "contradicts" else "",
                    reason=f"fixture {label}",
                ))
        return MemoryRelationResponse(decisions=decisions)

    def judge_change_impact(self, work, payload):
        return self.impact

    async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
        if response_format is SupportAssessmentWireResponse:
            self.support_requests.append(json.loads(prompt.split("<assessment>", 1)[1].split("</assessment>", 1)[0]))
        return await super().evaluate_revision_work(prompt, response_format=response_format, **kwargs)

    async def assess_support(self, prompt, **kwargs):
        from tests.revision_client_fixture import FixtureSupport

        payload = json.loads(prompt.split("<assessment>", 1)[1].split("</assessment>", 1)[0])
        claim = payload["claim"]
        rows = [row for row in payload["current"]["primary_candidates"] if row[0].startswith("PRM-")]
        wanted = self.support_text.get(claim, claim)
        row = next((row for row in rows if row[1] == wanted), rows[0])
        scripted = self.verdicts.get(claim)
        if scripted:
            # A scripted read keeps reading until the last group, which takes the next verdict.
            supported = payload["last"] and scripted.pop(0)
        else:
            supported = row[1] == wanted
        return FixtureSupport(status="supported" if supported else "unsupported", primary_ref=row[0], required_refs=[])

    def supplied_texts(self, request: dict) -> list[str]:
        return [row[1] for row in request["current"]["primary_candidates"]]


def _engine(db: Database, client) -> MemoryEngine:
    return MemoryEngine(
        cross_document_candidates=_candidate_retriever(build_sqlite_adapters(db, object())),
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=client,
    )


class _Page:
    """Successive revisions of one Confluence page, committed through the engine."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.current = None
        self.revision = 0

    def next(self, *paragraphs: str):
        self.revision += 1
        prior = self.current
        return _projection(
            run_id=f"projection-coordinator-{self.revision}",
            body="\n\n".join(paragraphs),
            prior=prior.source_unit_revisions[0] if prior else None,
            prior_observations=(
                {revision.observation_id: revision for revision in prior.observation_revisions} if prior else None
            ),
        )

    async def commit(self, client, projection, *claims: str):
        stats = await _engine(self.db, client).prepare_and_commit_projected_lifecycle(
            projection=projection,
            doc_id="confluence-123",
            raw_memories=_selected(projection, [RawMemory(content=claim, memory_type="fact") for claim in claims]),
            doc_type="design-doc",
            project_key="ENG",
            repo_identifier=None,
            document_content=projection.observation_revisions[-1].content,
            update_mode="full_document",
            changed_hunks=None,
            update_plan_stats=None,
            source_updated_at=datetime(2026, 7, 15 + self.revision, tzinfo=timezone.utc),
        )
        self.current = projection
        return stats

    @property
    def unit_id(self) -> str:
        return self.current.source_units[0].id


async def _seeded_page(db: Database, *paragraphs: str, claim: str = TWO):
    await db.enable_lifecycle_gate("src-1")
    page = _Page(db)
    await page.commit(_ScriptedClient(), page.next(*paragraphs), claim)
    [memory] = await db.list_memories()
    return page, memory


async def _reviews(db: Database, memory_id: str):
    return await db.list_lifecycle_reviews(incumbent_memory_ids=(memory_id,))


async def _support_texts(db: Database, memory_id: str) -> set[str | None]:
    return {part.excerpt for part in await active_support_evidence(db, memory_id, source_id="src-1")}


@pytest.mark.asyncio
async def test_a_contradicted_supported_claim_keeps_one_review_across_revisions(db: Database) -> None:
    page, memory = await _seeded_page(db, TWO, RETENTION)
    client = _ScriptedClient(relations={(ONE, TWO): "contradicts"})

    # The source now states both claims: the old one is rebound and the conflict goes to a Review.
    stats = await page.commit(client, page.next(TWO, ONE, RETENTION), ONE)
    [review] = await _reviews(db, memory.id)
    assert review.id == coordinator_review_id(page.unit_id, memory.id, ONE)
    assert review.status is LifecycleReviewStatus.PENDING
    assert review.staged_evidence["proposal"] == CoordinatorProposal.SUPERSEDE.value
    assert review.staged_evidence["stale_guard"]["support_set_hash"] == await db.get_memory_support_set_hash(memory.id)
    assert stats["coordinator_review_count"] == 1 and stats["pending_review"] == 1
    assert [item.id for item in await db.list_memories()] == [memory.id]
    rebound = await active_support_evidence(db, memory.id, source_id="src-1")
    assert {part.validation_unit_revision_id for part in rebound} == {page.current.source_unit_revisions[0].id}

    # An unrelated edit does not extract the Candidate again; its exact Evidence carries the conflict.
    stats = await page.commit(client, page.next(TWO, ONE, RETENTION, AUDIT))
    [carried] = await _reviews(db, memory.id)
    assert (carried.id, carried.status) == (review.id, LifecycleReviewStatus.PENDING)
    assert carried.lifecycle_plan_id != review.lifecycle_plan_id
    assert stats["coordinator_carried_conflict_count"] == 1 and stats["coordinator_recheck_count"] == 0
    assert [item.id for item in await db.list_memories()] == [memory.id]

    # The contradicting statement is gone, so the conflict is gone.
    await page.commit(client, page.next(TWO, RETENTION, AUDIT))
    [closed] = await _reviews(db, memory.id)
    assert (closed.id, closed.status) == (review.id, LifecycleReviewStatus.STALE)

    # It comes back and reopens the same Review, which a reviewer then approves.
    await page.commit(client, page.next(TWO, ONE, RETENTION, AUDIT), ONE)
    [reopened] = await _reviews(db, memory.id)
    assert (reopened.id, reopened.status, reopened.resolved_at) == (review.id, LifecycleReviewStatus.PENDING, None)
    payload = await db.get_lifecycle_plan_payload(reopened.lifecycle_plan_id)
    await db.apply_lifecycle_plan(build_lifecycle_review_approval_plan(reopened, payload))

    superseded = await db.get_memory(memory.id)
    assert superseded is not None and superseded.status == "superseded"
    assert [item.content for item in await db.list_memories() if item.status == "active"] == [ONE]


@pytest.mark.asyncio
async def test_a_rejected_conflict_is_not_raised_again(db: Database) -> None:
    page, memory = await _seeded_page(db, TWO, RETENTION)
    client = _ScriptedClient(relations={(ONE, TWO): "contradicts"})
    await page.commit(client, page.next(TWO, ONE, RETENTION), ONE)
    [review] = await _reviews(db, memory.id)
    await db.resolve_lifecycle_review(review.id, LifecycleReviewStatus.REJECTED, review_note="Two reviewers stand")

    stats = await page.commit(client, page.next(TWO, ONE + " ", RETENTION), ONE)

    [kept] = await _reviews(db, memory.id)
    assert (kept.id, kept.status) == (review.id, LifecycleReviewStatus.REJECTED)
    assert stats["pending_review"] == 0 and stats["added"] == 0
    assert [(item.id, item.status) for item in await db.list_memories()] == [(memory.id, "active")]


@pytest.mark.asyncio
async def test_an_unaffected_claim_contradicted_by_the_source_is_read_once_in_the_normal_order(db: Database) -> None:
    page, memory = await _seeded_page(db, TWO, RETENTION)
    client = _ScriptedClient(relations={(ONE, TWO): "contradicts"}, impact="unaffected")

    stats = await page.commit(client, page.next(TWO, RETENTION, ONE), ONE)

    assert stats["support_revalidation_change_impact_unaffected_count"] == 1
    assert stats["coordinator_recheck_count"] == 1
    # The re-check is an ordinary ordered read that may reach every group of the revision.
    [recheck] = client.support_requests
    assert TWO in client.supplied_texts(recheck)
    [review] = await _reviews(db, memory.id)
    assert review.staged_evidence["proposal"] == CoordinatorProposal.SUPERSEDE.value


@pytest.mark.asyncio
@pytest.mark.parametrize("recheck_supported", [True, False])
async def test_an_unsupported_claim_with_an_equivalent_is_rechecked_against_that_evidence(
    db: Database, recheck_supported: bool,
) -> None:
    page, memory = await _seeded_page(db, TWO, RETENTION)
    client = _ScriptedClient(
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
        assert await _reviews(db, memory.id) == []
        assert await _support_texts(db, memory.id) == {RESTATED}
        return
    [review] = await _reviews(db, memory.id)
    assert review.staged_evidence["proposal"] == CoordinatorProposal.REBIND.value
    assert review.staged_evidence["candidate"]["content"] == RESTATED
    assert await _support_texts(db, memory.id) == {TWO}
    payload = await db.get_lifecycle_plan_payload(review.lifecycle_plan_id)
    await db.apply_lifecycle_plan(build_lifecycle_review_approval_plan(review, payload))
    assert (await db.get_memory(memory.id)).status == "active"
    assert await _support_texts(db, memory.id) == {RESTATED}


@pytest.mark.asyncio
async def test_a_failed_recheck_leaves_the_revision_uncommitted_and_a_retry_keeps_the_candidate(db: Database) -> None:
    page, memory = await _seeded_page(db, TWO, RETENTION)

    class FailingRecheck(_ScriptedClient):
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
    assert await _reviews(db, memory.id) == []

    await page.commit(
        _ScriptedClient(relations=relations, support_text={TWO: RESTATED}, verdicts={TWO: [False, True]}),
        revision, RESTATED,
    )

    assert [item.id for item in await db.list_memories()] == [memory.id]
    assert await _support_texts(db, memory.id) == {RESTATED}


@pytest.mark.asyncio
async def test_rejecting_a_conflict_with_an_equivalent_binds_the_claim_to_that_equivalent(db: Database) -> None:
    page, memory = await _seeded_page(db, TWO, RETENTION)
    client = _ScriptedClient(
        relations={(RESTATED, TWO): "equivalent", (ONE, TWO): "contradicts"}, verdicts={TWO: [False]},
    )

    stats = await page.commit(client, page.next(RESTATED, ONE, RETENTION), RESTATED, ONE)

    assert stats["coordinator_recheck_count"] == 0
    [review] = await _reviews(db, memory.id)
    assert review.staged_evidence["candidate"]["content"] == ONE
    assert review.staged_evidence["rejection_candidate"]["content"] == RESTATED
    payload = await db.get_lifecycle_plan_payload(review.lifecycle_plan_id)
    rejection = build_lifecycle_review_rejection_plan(review, payload, review_note="Two reviewers stand")
    assert rejection is not None
    await db.apply_lifecycle_plan(rejection)

    [rejected] = await _reviews(db, memory.id)
    assert rejected.status is LifecycleReviewStatus.REJECTED
    assert (await db.get_memory(memory.id)).status == "active"
    assert await _support_texts(db, memory.id) == {RESTATED}


@pytest.mark.asyncio
async def test_support_and_relation_run_concurrently(db: Database) -> None:
    page, memory = await _seeded_page(db, TWO, RETENTION)
    relation_started, support_started = asyncio.Event(), asyncio.Event()

    class Rendezvous(_ScriptedClient):
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


async def _jira_incumbent(db: Database):
    """A claim whose Evidence is a comment the next, partial, page of comments leaves out."""
    await _set_fixture_source_type(db, "jira")
    await db.enable_lifecycle_gate("src-1")
    first = _jira_projection(
        run_id="projection-jira-coordinator-1", description="Initial issue description.",
        comment_id="501", comment_body="Decision: retain A7",
    )
    engine = _engine(db, _ScriptedClient())
    await engine.prepare_and_commit_projected_lifecycle(
        projection=first, doc_id="confluence-123",
        raw_memories=_selected(first, [RawMemory(
            content="A7 is retained.", memory_type="decision", evidence_quote="Decision: retain A7",
        )]),
        doc_type="ticket", project_key="ENG", repo_identifier=None, document_content="PAY-12",
        update_mode="diff_guided", changed_hunks=None, update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    [memory] = await db.list_memories()
    second = _jira_projection(
        run_id="projection-jira-coordinator-2", description="A7 stays for regular payroll.",
        comment_id="502", comment_body="Unrelated follow-up.", comments_truncated=True,
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    assert second.coverage.value == "partial_projection"
    return memory, first, second


async def _commit_jira(db: Database, client, base, projection, claim: str):
    return await _engine(db, client).prepare_and_commit_projected_lifecycle(
        projection=projection, doc_id="confluence-123",
        raw_memories=_selected(projection, [RawMemory(
            content=claim, memory_type="decision", evidence_quote="A7 stays for regular payroll.",
        )], base=base),
        doc_type="ticket", project_key="ENG", repo_identifier=None, document_content="PAY-12",
        update_mode="diff_guided", changed_hunks=None, update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_a_partial_coverage_equivalent_rebinds_the_claim_after_one_recheck(db: Database) -> None:
    memory, first, second = await _jira_incumbent(db)
    claim = "A7 stays for regular payroll."
    client = _ScriptedClient(
        relations={(claim, memory.content): "equivalent"}, support_text={memory.content: claim},
    )

    stats = await _commit_jira(db, client, first, second, claim)

    assert stats["support_revalidation_unresolved_partial_coverage_count"] == 1
    assert stats["coordinator_recheck_count"] == 1 and stats["added"] == 0
    [recheck] = client.support_requests
    assert client.supplied_texts(recheck) == [claim]
    # The Support that had an UNKNOWN part is replaced by the returned description.
    assert await _support_texts(db, memory.id) == {claim}
    assert await _reviews(db, memory.id) == []


@pytest.mark.asyncio
async def test_a_partial_coverage_contradiction_is_reviewed_and_never_superseded(db: Database) -> None:
    memory, first, second = await _jira_incumbent(db)
    claim = "A7 is dropped for regular payroll."
    client = _ScriptedClient(relations={(claim, memory.content): "contradicts"})
    before = await active_support_evidence(db, memory.id, source_id="src-1")

    stats = await _commit_jira(db, client, first, second, claim)

    assert stats["superseded"] == 0 and stats["coordinator_recheck_count"] == 0
    assert client.support_requests == []
    [review] = await _reviews(db, memory.id)
    assert review.staged_evidence["proposal"] == CoordinatorProposal.SUPERSEDE.value
    assert (await db.get_memory(memory.id)).status == "active"
    assert await active_support_evidence(db, memory.id, source_id="src-1") == before
