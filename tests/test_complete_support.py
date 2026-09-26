"""Complete Support: one definition, shared by Support Assessment and Candidate Admission.

A claim is completely supported only when every specific it states is in, or
follows directly from, its selected Evidence. Fixture judgments here apply that
definition to show what the pipeline supplies to the reading and what an
UNSUPPORTED answer does; they are not model-accuracy evidence.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from memforge.memory import candidate_admission
from memforge.memory.candidate_admission import CANDIDATE_ADMISSION_CONTRACT, admit_candidates
from memforge.models import RawMemory
from memforge.pipeline import revision_work
from memforge.pipeline.complete_support import COMPLETE_SUPPORT_DEFINITION
from memforge.pipeline.revision_assessment import REVISION_SUPPORT_CONTRACT, RevisionAssessmentContext
from memforge.pipeline.revision_work import SUPPORT_ASSESSMENT_CONTRACT, RevisionWorkExecutor
from memforge.source_representation import UNIT_TITLE_OBSERVATION_TYPE
from memforge.storage.database import Database
from tests.coordination_fixture import JIRA_DOCUMENT, SOURCE_ID, ScriptedClient, coordination_engine
from tests.revision_client_fixture import FixtureSupport
from tests.test_candidate_admission import AdmissionClient
from tests.test_projected_lifecycle_integration import (
    _jira_projection,
    _selected,
    _set_fixture_source_type,
    db as db,
)
from tests.test_revision_work import Client, work_items
from tests.unit_support_fixture import active_support_evidence

# The fixture issue is PAY-12 ("Payroll"); its comment states the decision.
DECISION = "Decision: retain A7"
TITLE_KEY = "PAY-12"
OTHER_KEY = "PAY-99"
ISSUE_KEY = re.compile(r"\b[A-Z][A-Z0-9]*-\d+\b")
FIRST_COMMIT_AT = datetime(2026, 9, 25, tzinfo=timezone.utc)


def test_complete_support_names_every_kind_of_specific_a_claim_can_state():
    for specific in (
        "names\nof people, systems and things", "identifiers", "quantities", "dates and times", "statuses",
        "conditions\nand scope",
    ):
        assert specific in COMPLETE_SUPPORT_DEFINITION
    assert "contradicts or does not contain is not\nsupported" in COMPLETE_SUPPORT_DEFINITION
    assert "even when the rest of the claim matches" in COMPLETE_SUPPORT_DEFINITION


def test_support_assessment_and_admission_share_the_one_definition_and_change_impact_does_not_use_it():
    assert revision_work.ASSESS_PROMPT.count(COMPLETE_SUPPORT_DEFINITION) == 1
    assert candidate_admission._ADMISSION_INSTRUCTIONS.count(COMPLETE_SUPPORT_DEFINITION) == 1
    assert COMPLETE_SUPPORT_DEFINITION not in revision_work.CHANGE_IMPACT_PROMPT


def test_the_definition_raises_every_contract_whose_result_it_defines():
    assert REVISION_SUPPORT_CONTRACT == "revision-support-v6"
    assert SUPPORT_ASSESSMENT_CONTRACT == "support-ordered-reading-v4"
    assert CANDIDATE_ADMISSION_CONTRACT == "candidate-admission-v2"


@pytest.mark.asyncio
async def test_every_support_reading_and_admission_request_carries_the_definition():
    client = Client(limit=16000)
    await RevisionWorkExecutor(client=client, model="fixture").assess_many(
        work_items("Two reviewers approve US releases.\n\nRelease notes are published weekly.\n"),
    )
    assert client.prompts and all(COMPLETE_SUPPORT_DEFINITION in prompt for prompt in client.prompts)
    assert all(COMPLETE_SUPPORT_DEFINITION not in prompt for prompt in client.impact_prompts)

    projection = _jira_projection(run_id="complete-support-admission", description="Payroll context.")
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
    catalog = context.catalog(context.full_fragments)
    decision = next(f for f in catalog.fragments if DECISION in f.presentation_text)
    admission = AdmissionClient()
    await admit_candidates([RawMemory(
        content=f"{TITLE_KEY} retains A7.", memory_type="decision",
        source_observation_id=decision.anchor.observation_id,
        resolved_evidence_selection=catalog.resolve_selection(primary_ref=decision.reference),
    )], client=admission, model="fixture")
    [prompt] = admission.prompts
    assert COMPLETE_SUPPORT_DEFINITION in prompt


class _IdentifierReadingClient(ScriptedClient):
    """Judges Support by the definition: every issue key the claim names must be the Unit Title's key."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.titles_read: list[str] = []

    async def assess_support(self, prompt, **kwargs):
        payload = json.loads(prompt.split("<assessment>", 1)[1].split("</assessment>", 1)[0])
        rows = [*payload["current"]["primary_candidates"], *payload["current"]["required_only_candidates"]]
        title = next((row for row in rows if row[1].startswith("Jira issue\n")), None)
        decision = next(row for row in rows if row[0].startswith("PRM-") and DECISION in row[1])
        if title is None:
            return FixtureSupport(status="unsupported", primary_ref=decision[0])
        self.titles_read.append(title[1])
        [title_key] = re.findall(r"^Key: (\S+)$", title[1], flags=re.MULTILINE)
        named = set(ISSUE_KEY.findall(payload["claim"]))
        status = "supported" if named == {title_key} else "unsupported"
        return FixtureSupport(status=status, primary_ref=decision[0], required_refs=[title[0]])


def _title_observation_id(projection) -> str:
    return next(o.id for o in projection.observations if o.observation_type == UNIT_TITLE_OBSERVATION_TYPE)


def _claim(projection, content: str) -> RawMemory:
    """A claim whose Evidence is the decision comment with the Unit Title as Required Evidence."""
    return RawMemory(
        content=content, memory_type="decision", evidence_quote=DECISION,
        required_source_observation_ids=[_title_observation_id(projection)],
    )


async def _commit(db: Database, client, projection, claims, *, day: int, **options):
    return await coordination_engine(db, client).prepare_and_commit_projected_lifecycle(
        projection=projection, doc_id="confluence-123", raw_memories=_selected(projection, claims),
        doc_type="ticket", project_key="ENG", repo_identifier=None, document_content=JIRA_DOCUMENT,
        update_mode="diff_guided", changed_hunks=None, update_plan_stats=None,
        source_updated_at=FIRST_COMMIT_AT + timedelta(days=day), **options,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("correction", [False, True], ids=["retired", "superseded"])
async def test_a_reprocessed_claim_naming_another_identifier_than_its_unit_title_is_unsupported(
    db: Database, correction: bool,
) -> None:
    await _set_fixture_source_type(db, "jira")
    await db.enable_lifecycle_gate(SOURCE_ID)
    first = _jira_projection(
        run_id="complete-support-1", description="Payroll context.", comment_id="501", comment_body=DECISION,
    )
    wrong = f"{OTHER_KEY} retains A7."
    await _commit(db, ScriptedClient(), first, [_claim(first, wrong)], day=0)
    [memory] = await db.list_memories()
    assert memory.content == wrong

    # An operator reprocess reads the whole Unit at its current revision, with no usable baseline.
    again = _jira_projection(
        run_id="complete-support-again", description="Payroll context.", comment_id="501", comment_body=DECISION,
        prior=first.source_unit_revisions[0],
        prior_observations={revision.observation_id: revision for revision in first.observation_revisions},
    )
    corrected = f"{TITLE_KEY} retains A7."
    client = _IdentifierReadingClient(relations={(corrected, wrong): "contradicts"})
    stats = await _commit(
        db, client, again, [_claim(again, corrected)] if correction else [],
        day=1, derivation_support_without_baseline=True,
    )

    assert stats["support_revalidation_reprocess_count"] == 1
    # The reading supplied the Unit Title, whose key contradicts the key the claim names.
    assert client.titles_read and all(f"Key: {TITLE_KEY}" in title for title in client.titles_read)
    old = await db.get_memory(memory.id)
    if correction:
        assert stats["superseded"] == 1 and old.status == "superseded"
        replacement = await db.get_memory(old.superseded_by)
        assert replacement.content == corrected and replacement.status == "active"
    else:
        assert old.status == "retired"
        assert await active_support_evidence(db, memory.id, source_id=SOURCE_ID) == ()
