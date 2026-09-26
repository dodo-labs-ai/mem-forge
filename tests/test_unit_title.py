"""The Unit Title: every adapter names its Unit, and every reading of the Unit reads that name."""

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.llm.structured import ChangeImpactWireResponse
from memforge.memory.candidate_admission import admit_candidates
from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole
from memforge.models import ContentItem, NormalizedContent, RawContent, RawMemory
from memforge.pipeline.evidence_fragments import build_revision_fragment_index
from memforge.pipeline.extraction_requests import plan_extraction_requests
from memforge.pipeline.projection_context import (
    CommittedSourceUnitSnapshot,
    ExtractionAuthority,
    plan_projection_evidence_work,
)
from memforge.pipeline.projection_evidence import build_projected_claim_evidence
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.revision_work import RevisionWorkExecutor
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.pipeline.support_reading import (
    EvidenceCorrespondence,
    SupportRoute,
    SupportWorkItem,
    plan_support_revision,
)
from memforge.source_projection import ProjectionCoverage
from memforge.source_representation import UNIT_TITLE_OBSERVATION_TYPE
from tests.llm_fixture import NoopMemoryExtractor
from tests.revision_client_fixture import change_impact_response, continued, supported
from tests.test_candidate_admission import AdmissionClient
from tests.test_evidence_catalog_source_matrix import TEXTUAL_BUILTIN_SOURCE_TYPES, _projection_inputs
from tests.test_revision_assessment import memory
from tests.test_revision_work import Client

COMMENT = "Decision: retain A7 for regular payroll."
NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def jira(summary="Payroll run fails", *, issue_type=None, description="Payroll context.", comments_truncated=False,
         prior=None, run_id="run-1"):
    item = ContentItem(
        item_id="jira-SFPAY-180000", title=summary, source_url="https://jira.example.test/browse/SFPAY-180000",
        last_modified=NOW, version=run_id, extra={"issue_key": "SFPAY-180000"},
    )
    fields = {
        "summary": summary, "description": description, "status": None, "priority": None,
        "assignee": None, "labels": [], "resolution": None, "updated": NOW.isoformat(),
        **({"issuetype": {"name": issue_type}} if issue_type else {}),
    }
    payload = {
        "id": "180000", "key": "SFPAY-180000", "fields": fields,
        "_comments": [{"id": "501", "body": COMMENT}], "_comments_included": True,
        "_comments_total": 2 if comments_truncated else 1,
        "changelog": {"startAt": 0, "histories": [], "total": 0},
        **({"_comments_truncated": {"returned": 1, "total": 2}} if comments_truncated else {}),
    }
    return project_source_item(
        source_id="src-jira", source_type="jira", run_id=run_id, item=item,
        raw=RawContent(item=item, body=json.dumps(payload).encode(), content_type="application/json"),
        normalized=NormalizedContent(item=item, markdown_body="normalized Jira"),
        prior_unit_revision=prior.source_unit_revisions[0] if prior else None,
        prior_observation_revisions=(
            {revision.observation_id: revision for revision in prior.observation_revisions} if prior else None
        ),
    )


def title_revision(projection):
    [observation] = [item for item in projection.observations if item.observation_type == UNIT_TITLE_OBSERVATION_TYPE]
    return next(item for item in projection.observation_revisions if item.observation_id == observation.id)


def fragment(projection, text, *, base=None):
    context = RevisionAssessmentContext(projection=projection, base=base, access_context_hash="scope")
    return next(f for f in context.full_fragments if text in f.presentation_text)


def support_part(projection, text, *, role=EvidenceRole.PRIMARY, reference_id="e1"):
    found = fragment(projection, text)
    return ActiveSupportEvidence(
        memory_id="memory", source_id=projection.source_id, reference_id=reference_id, evidence_unit_id="eu1",
        role=role, anchor=found.anchor, excerpt=found.presentation_text,
        raw_content_sha256=found.raw_content_sha256, presentation_sha256=found.presentation_sha256,
    )


def plan_supports(context, *supports):
    items = [SupportWorkItem(f"w{index}", memory(), support, context) for index, support in enumerate(supports)]
    return plan_support_revision(context, items)


def extraction_requests(projection, authority, *, base=None):
    return plan_extraction_requests(
        RevisionAssessmentContext(projection=projection, base=base, access_context_hash="scope"),
        authority, extractor=NoopMemoryExtractor(), source_type=projection.source_type, doc_type="ticket",
    ).requests


def test_jira_unit_title_names_the_issue_from_payload_values_only():
    assert title_revision(jira(issue_type="Defect")).content == (
        "Jira issue\nKey: SFPAY-180000\nType: Defect\nSummary: Payroll run fails"
    )
    # A package without an issue type names what it has; nothing is guessed.
    assert title_revision(jira()).content == "Jira issue\nKey: SFPAY-180000\nSummary: Payroll run fails"


@pytest.mark.parametrize("source_type", [*TEXTUAL_BUILTIN_SOURCE_TYPES, "extension_document"])
def test_every_adapter_projects_its_unit_title_first(source_type):
    item, raw, normalized, _ = _projection_inputs(source_type, "Use traceId.", version="1")
    projection = project_source_item(
        source_id=f"src-{source_type}", source_type=source_type, run_id="run-title",
        item=item, raw=raw, normalized=normalized,
    )
    first = projection.observations[0]
    title = title_revision(projection).content

    assert first.observation_type == UNIT_TITLE_OBSERVATION_TYPE
    assert [item.observation_type for item in projection.observations].count(UNIT_TITLE_OBSERVATION_TYPE) == 1
    kind, *fields = title.split("\n")
    assert kind and all(": " in field and field.split(": ", 1)[1] for field in fields)
    assert "None" not in title


def test_unit_title_is_one_required_only_fragment():
    revision = title_revision(jira())
    [only] = build_revision_fragment_index(revision).fragments

    assert only.primary_eligible is False
    assert only.fragment_type == "unit-identity"
    assert only.presentation_text == revision.content


@pytest.mark.parametrize("transition", ["initial", "update", "reprocess"])
def test_unit_title_is_never_primary_but_is_read_with_every_extraction_request(transition):
    first = jira()
    if transition == "initial":
        projection, base, committed, reprocess = first, None, None, False
    else:
        projection = jira(description="Payroll context changed.", prior=first, run_id="run-2")
        base, committed = first, CommittedSourceUnitSnapshot(
            unit_revision=first.source_unit_revisions[0], observation_revisions=first.observation_revisions,
        )
        reprocess = transition == "reprocess"
    authority = plan_projection_evidence_work(
        projection, committed_base_snapshot=committed, reprocess_all_current_observations=reprocess,
    )
    assert isinstance(authority, ExtractionAuthority)
    requests = extraction_requests(projection, authority, base=base)
    title = title_revision(projection)

    assert requests
    for request in requests:
        [read_title] = [f for f in request.catalog.fragments if f.anchor.observation_id == title.observation_id]
        assert read_title.primary_eligible is False
        assert read_title.presentation_text == title.content


def test_unit_title_is_selectable_as_required_and_persisted_as_evidence():
    projection = jira(issue_type="Defect")
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
    catalog = context.catalog(context.full_fragments)
    comment = next(f for f in catalog.fragments if COMMENT in f.presentation_text)
    title = next(f for f in catalog.fragments if f.fragment_type == "unit-identity")
    selection = catalog.resolve_selection(primary_ref=comment.reference, required_refs=[title.reference])
    raw = RawMemory(
        content="SFPAY-180000 retains A7 for regular payroll.", memory_type="decision",
        source_observation_id=comment.anchor.observation_id, resolved_evidence_selection=selection,
    )

    evidence = build_projected_claim_evidence(
        projection=projection, raw_memories=(raw,), doc_id="jira-SFPAY-180000", source_type="jira",
        project_key=None, visibility="workspace", owner_user_id=None, repo_identifier=None,
        access_context_hash="scope", extractor_run_id="run-1", observed_at=NOW.isoformat(),
    )

    required = [item for item in evidence.references if item.role is EvidenceRole.REQUIRED]
    assert [item.anchor.observation_id for item in required] == [title.anchor.observation_id]
    assert required[0].excerpt == title.presentation_text


@pytest.mark.asyncio
async def test_admission_reads_the_unit_title_only_when_the_candidate_selected_it():
    projection = jira(issue_type="Defect")
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
    catalog = context.catalog(context.full_fragments)
    comment = next(f for f in catalog.fragments if COMMENT in f.presentation_text)
    title = next(f for f in catalog.fragments if f.fragment_type == "unit-identity")

    def candidate(content, required):
        return RawMemory(
            content=content, memory_type="decision", source_observation_id=comment.anchor.observation_id,
            resolved_evidence_selection=catalog.resolve_selection(primary_ref=comment.reference, required_refs=required),
        )

    named = candidate("SFPAY-180000 retains A7 for regular payroll.", [title.reference])
    unnamed = candidate("SFPAY-174295 retains A7 for regular payroll.", [])
    client = AdmissionClient()

    await admit_candidates([named, unnamed], client=client, model="fixture")

    [request] = client.requests
    evidence = {
        row["claim"]: [request["evidence_catalog"][ref]["excerpt"] for ref in row["evidence_refs"]]
        for row in request["candidates"]
    }
    # The key the claim names is Evidence only where the Candidate selected the Unit Title.
    assert title.presentation_text in evidence[named.content]
    assert all("SFPAY-180000" not in excerpt for excerpt in evidence[unnamed.content])


def test_a_unit_first_projected_with_its_title_sends_exact_supports_to_change_impact_without_extraction():
    current = jira(run_id="run-legacy")
    title = title_revision(current)
    legacy_revisions = tuple(item for item in current.observation_revisions if item.id != title.id)
    legacy_unit = replace(
        current.source_unit_revisions[0], id="unitrev-legacy", semantic_hash="legacy",
        observation_revision_ids=tuple(sorted(item.id for item in legacy_revisions)),
    )
    legacy = replace(
        current,
        observations=tuple(item for item in current.observations if item.id != title.observation_id),
        observation_revisions=legacy_revisions,
        source_unit_revisions=(legacy_unit,),
        deltas=(replace(
            current.deltas[0],
            current_unit_revision_id=legacy_unit.id,
            changed_anchors=tuple(a for a in current.deltas[0].changed_anchors if a.observation_id != title.observation_id),
            added_observation_ids=tuple(i for i in current.deltas[0].added_observation_ids if i != title.observation_id),
        ),),
    )
    upgraded = project_source_item(
        source_id="src-jira", source_type="jira", run_id="run-upgraded",
        item=ContentItem(item_id="jira-SFPAY-180000", title="Payroll run fails", source_url="", last_modified=NOW),
        raw=RawContent(
            item=ContentItem(item_id="jira-SFPAY-180000", title="", source_url="", last_modified=NOW),
            body=json.dumps({
                "id": "180000", "key": "SFPAY-180000",
                "fields": {"summary": "Payroll run fails", "description": "Payroll context.", "status": None,
                           "priority": None, "assignee": None, "labels": [], "resolution": None,
                           "updated": NOW.isoformat()},
                "_comments": [{"id": "501", "body": COMMENT}], "_comments_included": True, "_comments_total": 1,
                "changelog": {"startAt": 0, "histories": [], "total": 0},
            }).encode(),
            content_type="application/json",
        ),
        normalized=NormalizedContent(
            item=ContentItem(item_id="jira-SFPAY-180000", title="", source_url="", last_modified=NOW),
            markdown_body="normalized Jira",
        ),
        prior_unit_revision=legacy_unit,
        prior_observation_revisions={item.observation_id: item for item in legacy_revisions},
    )
    assert upgraded.deltas[0].added_observation_ids == (title.observation_id,)

    authority = plan_projection_evidence_work(
        upgraded,
        committed_base_snapshot=CommittedSourceUnitSnapshot(unit_revision=legacy_unit, observation_revisions=legacy_revisions),
        reprocess_all_current_observations=False,
    )
    assert isinstance(authority, ExtractionAuthority)
    # The title is the only change, and it authorizes no extraction.
    assert extraction_requests(upgraded, authority, base=legacy) == ()

    context = RevisionAssessmentContext(projection=upgraded, base=legacy, access_context_hash="scope")
    revision_plan = plan_supports(context, (support_part(legacy, COMMENT),))
    [support] = revision_plan.supports
    assert [part.status for part in support.parts] == [EvidenceCorrespondence.EXACT_UNCHANGED]
    assert support.route is SupportRoute.CHANGE_IMPACT
    title_ref = next(f.reference for f in revision_plan.catalog.fragments if f.fragment_type == "unit-identity")
    assert revision_plan.changed_refs == frozenset({title_ref})


def test_a_summary_change_modifies_the_supports_that_selected_the_title_and_asks_change_impact_for_the_rest():
    first = jira()
    renamed = jira("Payroll run fails for off-cycle runs", prior=first, run_id="run-2")
    context = RevisionAssessmentContext(projection=renamed, base=first, access_context_hash="scope")
    named = (
        support_part(first, COMMENT),
        support_part(first, "Jira issue", role=EvidenceRole.REQUIRED, reference_id="e2"),
    )
    unnamed = (support_part(first, COMMENT),)

    with_title, without_title = plan_supports(context, named, unnamed).supports

    assert [part.status for part in with_title.parts] == [
        EvidenceCorrespondence.EXACT_UNCHANGED, EvidenceCorrespondence.MODIFIED,
    ]
    assert with_title.route is SupportRoute.SUPPORT_ASSESSMENT
    assert without_title.route is SupportRoute.CHANGE_IMPACT


def test_partial_projection_always_returns_the_unit_title():
    first = jira()
    partial = jira(comments_truncated=True, prior=first, run_id="run-partial")
    assert partial.coverage is ProjectionCoverage.PARTIAL_PROJECTION
    context = RevisionAssessmentContext(projection=partial, base=first, access_context_hash="scope")

    [support] = plan_supports(
        context, (support_part(first, "Jira issue", role=EvidenceRole.REQUIRED),),
    ).supports

    assert [part.status for part in support.parts] == [EvidenceCorrespondence.EXACT_UNCHANGED]


@pytest.mark.asyncio
async def test_the_unit_title_is_read_in_every_support_step_and_change_impact_request():
    first = jira()
    changed = jira(description="Payroll context changed.", prior=first, run_id="run-2")
    context = RevisionAssessmentContext(projection=changed, base=first, access_context_hash="scope")
    title = title_revision(changed).content

    class ReadingClient(Client):
        def judge(self, prompt):
            data = json.loads(prompt.split("<assessment>")[1].split("</assessment>")[0])
            rows = [*data["current"]["primary_candidates"], *data["current"]["required_only_candidates"]]
            comment = next((row[0] for row in rows if row[1] == COMMENT), None)
            return [
                supported(work, comment) if comment and work["may_conclude"] else continued(work)
                for work in data["works"]
            ]

        async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
            if response_format is ChangeImpactWireResponse:
                self.impact_prompts.append(prompt)
                return change_impact_response(prompt)
            return await super().evaluate_revision_work(prompt, response_format=response_format, **kwargs)

    client = ReadingClient(limit=40000)
    items = [SupportWorkItem("w0", memory(), (support_part(first, COMMENT),), context)]
    [result] = (await RevisionWorkExecutor(client=client, model="fixture").assess_many(items)).values()

    assert result.supported
    assert client.impact_prompts and client.prompts
    assert all(json.dumps(title)[1:-1] in prompt for prompt in (*client.impact_prompts, *client.prompts))


@pytest.mark.parametrize("with_base", [True, False], ids=["update", "no-baseline"])
def test_the_unit_title_is_never_a_support_reading_step(with_base):
    first = jira()
    changed = jira(summary="Payroll run fails on A7", description="Payroll context changed.", prior=first, run_id="run-2")
    context = RevisionAssessmentContext(projection=changed, base=first if with_base else None, access_context_hash="scope")
    title = fragment(changed, "Jira issue", base=first if with_base else None)

    plan = plan_supports(context, (support_part(first, COMMENT),))
    order = plan.reading_order(plan.supports)

    def reads_title(fragments):
        return any(f.anchor == title.anchor for f in fragments)

    assert not any(reads_title(group) for group in plan.groups)
    assert title.anchor in context.reading_context(plan.groups[0])
    title_steps = [part for part in order.parts if reads_title(part.fragments)]
    # A changed Unit Title is one of the changes read first; it is never a step of the document order.
    assert len(title_steps) == (1 if with_base else 0)
    assert all(part in plan.changes for part in title_steps)
