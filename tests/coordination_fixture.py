"""Drive SupportRelationCoordinator end to end on SQLite through successive Source Unit revisions.

``ScriptedClient`` states Relation labels per (Candidate, old claim) pair and
Support per claim; ``Page`` commits successive revisions of one Confluence page;
``seed_jira_issue`` and ``partial_jira_revision`` build a Jira issue whose next
revision is a partial page of comments.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from memforge.llm.structured import (
    MemoryRelationDecision,
    MemoryRelationResponse,
    SupportAssessmentWireResponse,
)
from memforge.memory.engine import MemoryEngine
from memforge.models import RawMemory
from memforge.source_projection import SourceProjection
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database
from tests.revision_client_fixture import FixtureSupport, RevisionClientFixture
from tests.test_projected_lifecycle_integration import (
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
SOURCE_ID = "src-1"
DOC_ID = "confluence-123"
JIRA_DOCUMENT = "PAY-12"
FIRST_REVISION_AT = datetime(2026, 7, 15, tzinfo=timezone.utc)


class ScriptedClient(RevisionClientFixture):
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


def coordination_engine(db: Database, client, memory_store=None) -> MemoryEngine:
    return MemoryEngine(
        cross_document_candidates=_candidate_retriever(build_sqlite_adapters(db, object())),
        db=db,
        memory_store=memory_store or _OutboxDrainer(db),
        structured_llm_client=client,
    )


class Page:
    """Successive revisions of one Confluence page, committed through the engine."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.current = None
        self.revision = 0

    def next(self, *paragraphs: str) -> SourceProjection:
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

    async def commit(self, client, projection: SourceProjection, *claims: str):
        stats = await coordination_engine(self.db, client).prepare_and_commit_projected_lifecycle(
            projection=projection,
            doc_id=DOC_ID,
            raw_memories=_selected(projection, [RawMemory(content=claim, memory_type="fact") for claim in claims]),
            doc_type="design-doc",
            project_key="ENG",
            repo_identifier=None,
            document_content=projection.observation_revisions[-1].content,
            update_mode="full_document",
            changed_hunks=None,
            update_plan_stats=None,
            source_updated_at=FIRST_REVISION_AT + timedelta(days=self.revision),
        )
        self.current = projection
        return stats

    @property
    def unit_id(self) -> str:
        return self.current.source_units[0].id


async def seeded_page(db: Database, *paragraphs: str, claim: str = TWO):
    """A page whose first revision states ``paragraphs`` and commits one Memory of ``claim``."""
    await db.enable_lifecycle_gate(SOURCE_ID)
    page = Page(db)
    await page.commit(ScriptedClient(), page.next(*paragraphs), claim)
    [memory] = await db.list_memories()
    return page, memory


async def lifecycle_reviews(db: Database, memory_id: str):
    return await db.list_lifecycle_reviews(incumbent_memory_ids=(memory_id,))


async def support_texts(db: Database, memory_id: str) -> set[str | None]:
    return {part.excerpt for part in await active_support_evidence(db, memory_id, source_id=SOURCE_ID)}


async def seed_jira_issue(db: Database, *, run_id: str, description: str, claims: Sequence[RawMemory]):
    """Commit an issue whose description and comment 501 ("Decision: retain A7") carry ``claims``."""
    await _set_fixture_source_type(db, "jira")
    await db.enable_lifecycle_gate(SOURCE_ID)
    first = _jira_projection(
        run_id=run_id, description=description, comment_id="501", comment_body="Decision: retain A7",
    )
    await coordination_engine(db, ScriptedClient()).prepare_and_commit_projected_lifecycle(
        projection=first, doc_id=DOC_ID, raw_memories=_selected(first, list(claims)),
        doc_type="ticket", project_key="ENG", repo_identifier=None, document_content=JIRA_DOCUMENT,
        update_mode="diff_guided", changed_hunks=None, update_plan_stats=None,
        source_updated_at=FIRST_REVISION_AT,
    )
    return first


def partial_jira_revision(first: SourceProjection, *, run_id: str, description: str) -> SourceProjection:
    """The next revision: a new description and a partial page of comments that leaves 501 out."""
    second = _jira_projection(
        run_id=run_id, description=description,
        comment_id="502", comment_body="Unrelated follow-up.", comments_truncated=True,
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    assert second.coverage.value == "partial_projection"
    return second
