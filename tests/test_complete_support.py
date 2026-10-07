"""Whole Support uses immutable source framing; admission judges value and duplicates.

Fixture judgments test supplied evidence and lifecycle consequences, not model accuracy.
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
from memforge.pipeline.unit_title import UNIT_TITLE_DEFINITION, render_unit_title
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


def test_whole_support_preserves_material_claim_qualifications_and_source_scope():
    definition = " ".join(COMPLETE_SUPPORT_DEFINITION.split())
    for specific in ("conditions", "exceptions", "scope", "modality", "attribution", "effective boundaries"):
        assert specific in definition
    assert "changes outside selected refs" in definition
    assert "omission alone does not prove" in definition


def test_display_title_cannot_substantiate_claim_facts():
    assert "display-only" in COMPLETE_SUPPORT_DEFINITION
    assert "not source Evidence" in UNIT_TITLE_DEFINITION
    assert "cannot substantiate claim facts" in UNIT_TITLE_DEFINITION
    assert "authenticated source fact or referent" in COMPLETE_SUPPORT_DEFINITION


def test_support_assessment_uses_whole_support_and_admission_has_no_entailment_repair():
    assert revision_work.ASSESS_PROMPT.count(COMPLETE_SUPPORT_DEFINITION) == 1
    assert COMPLETE_SUPPORT_DEFINITION not in candidate_admission._ADMISSION_INSTRUCTIONS
    assert "Same-round duplicates" in candidate_admission._ADMISSION_INSTRUCTIONS
    assert "Never rewrite or merge claim text" in candidate_admission._ADMISSION_INSTRUCTIONS
    assert COMPLETE_SUPPORT_DEFINITION not in revision_work.CHANGE_IMPACT_PROMPT


def test_the_definition_raises_every_contract_whose_result_it_defines():
    assert REVISION_SUPPORT_CONTRACT == "revision-support-v11"
    assert SUPPORT_ASSESSMENT_CONTRACT == "support-ordered-reading-v9"
    assert CANDIDATE_ADMISSION_CONTRACT == "candidate-admission-v7"


@pytest.mark.asyncio
async def test_support_reads_full_contract_and_value_admission_receives_authenticated_source():
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
    )], client=admission, model="fixture", unit_title=projection.unit_title)
    [prompt] = admission.prompts
    assert COMPLETE_SUPPORT_DEFINITION not in prompt
    assert f"Issue: {TITLE_KEY}" in prompt
    assert f"<unit_title>\n{render_unit_title(projection.unit_title)}\n</unit_title>" in prompt


class _IdentifierReadingClient(ScriptedClient):
    """Fixture judgment reads the immutable comment issue key, independent of display aliases."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.source_keys_read: list[set[str]] = []

    async def assess_support(self, prompt, **kwargs):
        payload = json.loads(prompt.split("<assessment>", 1)[1].split("</assessment>", 1)[0])
        rows = [*payload["current"]["primary_candidates"], *payload["current"]["required_only_candidates"]]
        decision = next(row for row in rows if row[0].startswith("PRM-") and DECISION in row[1])
        source_keys = set(ISSUE_KEY.findall(decision[1]))
        self.source_keys_read.append(source_keys)
        named = set(ISSUE_KEY.findall(payload["claim"]))
        status = "supported" if named == source_keys else "unsupported"
        return FixtureSupport(status=status, primary_ref=decision[0])


def _claim(content: str) -> RawMemory:
    """A claim whose Evidence is an issue-framed immutable decision comment."""
    return RawMemory(content=content, memory_type="decision", evidence_quote=DECISION)


async def _commit(db: Database, client, projection, claims, *, day: int, **options):
    return await coordination_engine(db, client).prepare_and_commit_projected_lifecycle(
        projection=projection, doc_id="confluence-123", raw_memories=_selected(projection, claims),
        doc_type="ticket", project_key="ENG", repo_identifier=None, document_content=JIRA_DOCUMENT,
        update_mode="diff_guided", changed_hunks=None, update_plan_stats=None,
        source_updated_at=FIRST_COMMIT_AT + timedelta(days=day), **options,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("correction", [False, True], ids=["retired", "superseded"])
async def test_reprocessed_claim_naming_another_identifier_than_immutable_source_is_unsupported(
    db: Database, correction: bool,
) -> None:
    await _set_fixture_source_type(db, "jira")
    await db.enable_lifecycle_gate(SOURCE_ID)
    first = _jira_projection(
        run_id="complete-support-1", description="Payroll context.", comment_id="501", comment_body=DECISION,
    )
    wrong = f"{OTHER_KEY} retains A7."
    await _commit(db, ScriptedClient(), first, [_claim(wrong)], day=0)
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
        db, client, again, [_claim(corrected)] if correction else [],
        day=1, derivation_support_without_baseline=True,
    )

    assert stats["support_revalidation_reprocess_count"] == 1
    assert client.source_keys_read and all(keys == {TITLE_KEY} for keys in client.source_keys_read)
    old = await db.get_memory(memory.id)
    if correction:
        assert stats["superseded"] == 1 and old.status == "superseded"
        replacement = await db.get_memory(old.superseded_by)
        assert replacement.content == corrected and replacement.status == "active"
    else:
        assert old.status == "retired"
        assert await active_support_evidence(db, memory.id, source_id=SOURCE_ID) == ()
