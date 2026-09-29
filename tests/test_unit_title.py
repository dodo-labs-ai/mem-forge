"""The Unit Title: every adapter names its Unit, and every model reading of the Unit shows that name.

The Unit Title is reading context carried by the projection. It is no Observation,
no Evidence and no part of a Unit revision, so revisions are compared in the
current representation only.
"""

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.llm.structured import ChangeImpactWireResponse
from memforge.memory.candidate_admission import admit_candidates
from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole
from memforge.models import ContentItem, NormalizedContent, RawContent, RawMemory
from memforge.pipeline.extraction_requests import plan_extraction_requests
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.projection_context import (
    CommittedSourceUnitSnapshot,
    ExtractionAuthority,
    plan_projection_evidence_work,
)
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.revision_work import RevisionWorkExecutor
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.pipeline.support_reading import (
    EvidenceCorrespondence,
    SupportRoute,
    SupportWorkItem,
    plan_support_revision,
)
from memforge.pipeline.unit_title import unit_title_block
from memforge.source_projection import (
    AnchorKind,
    EvidenceCoordinateSpace,
    EvidenceRepresentationProfile,
    ProjectionCoverage,
    SourceAnchor,
    SourceObservation,
    SourceObservationRevision,
    UnitTitle,
    source_projection_from_payload,
    source_projection_to_payload,
)
from memforge.source_representation import in_current_representation
from memforge.storage import database as database_module
from memforge.storage.database import Database
from tests.llm_fixture import NoopMemoryExtractor
from tests.revision_client_fixture import change_impact_response, continued, supported
from tests.test_candidate_admission import AdmissionClient
from tests.test_evidence_catalog_source_matrix import TEXTUAL_BUILTIN_SOURCE_TYPES, _projection_inputs
from tests.test_revision_assessment import memory
from tests.test_revision_work import Client

COMMENT = "Decision: retain A7 for regular payroll."
NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)
# How a Unit Title was stored when it was the Unit's first Observation.
LEGACY_TITLE_TYPE = "unit_identity"
LEGACY_TITLE_PROFILE = EvidenceRepresentationProfile(
    name="unit-identity", version=1, coordinate_space=EvidenceCoordinateSpace.UNICODE_SCALAR,
)


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


def with_stored_title(projection):
    """The projection as stored when its Unit Title was the Unit's first Observation."""
    unit = projection.source_units[0]
    observation = SourceObservation(
        id="obs-stored-title", source_id=projection.source_id, source_unit_id=unit.id,
        observation_type=LEGACY_TITLE_TYPE, provider_key="$unit_identity", locator={},
    )
    revision = SourceObservationRevision(
        id="obsrev-stored-title", observation_id=observation.id, semantic_hash="stored-title",
        content=projection.unit_title.text, metadata={"provider_key": "$unit_identity"},
        evidence_profile=LEGACY_TITLE_PROFILE,
    )
    unit_revision = replace(
        projection.source_unit_revisions[0], id="unitrev-stored-title", semantic_hash="stored-title",
        membership_hash="stored-title",
        observation_revision_ids=tuple(sorted((*projection.source_unit_revisions[0].observation_revision_ids, revision.id))),
    )
    return replace(
        projection,
        run_id=f"{projection.run_id}-stored-title",
        observations=(observation, *projection.observations),
        observation_revisions=(revision, *projection.observation_revisions),
        source_unit_revisions=(unit_revision,),
        deltas=(replace(projection.deltas[0], current_unit_revision_id=unit_revision.id),),
    )


def title_part(stored):
    """Required Evidence a Support selected on the stored Unit Title."""
    revision = next(item for item in stored.observation_revisions if item.evidence_profile == LEGACY_TITLE_PROFILE)
    return ActiveSupportEvidence(
        memory_id="memory", source_id=stored.source_id, reference_id="e-title", evidence_unit_id="eu1",
        role=EvidenceRole.REQUIRED,
        anchor=SourceAnchor(
            kind=AnchorKind.REVISION_RANGE, observation_id=revision.observation_id,
            observation_revision_id=revision.id, range_start=0, range_end=len(revision.content),
        ),
        excerpt=revision.content, raw_content_sha256="stored-title", presentation_sha256="stored-title",
    )


def support_part(projection, text, *, role=EvidenceRole.PRIMARY, reference_id="e1"):
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
    found = next(f for f in context.full_fragments if text in f.presentation_text)
    return ActiveSupportEvidence(
        memory_id="memory", source_id=projection.source_id, reference_id=reference_id, evidence_unit_id="eu1",
        role=role, anchor=found.anchor, excerpt=found.presentation_text,
        raw_content_sha256=found.raw_content_sha256, presentation_sha256=found.presentation_sha256,
    )


def plan_supports(context, *supports):
    items = [SupportWorkItem(f"w{index}", memory(), support, context) for index, support in enumerate(supports)]
    return plan_support_revision(context, items)


def committed(projection):
    return CommittedSourceUnitSnapshot(
        unit_revision=projection.source_unit_revisions[0], observation_revisions=projection.observation_revisions,
    )


class NoModelClient(Client):
    async def evaluate_revision_work(self, prompt, *, response_format, **kwargs):
        raise AssertionError("a pure rebind calls no model")


def test_jira_unit_title_names_the_issue_from_payload_values_only():
    assert jira(issue_type="Defect").unit_title.text == (
        "Jira issue\nKey: SFPAY-180000\nType: Defect\nSummary: Payroll run fails"
    )
    # A package without an issue type names what it has; nothing is guessed.
    assert jira().unit_title.text == "Jira issue\nKey: SFPAY-180000\nSummary: Payroll run fails"


@pytest.mark.parametrize("source_type", [*TEXTUAL_BUILTIN_SOURCE_TYPES, "extension_document"])
def test_every_adapter_names_its_unit_outside_its_observations(source_type):
    item, raw, normalized, _ = _projection_inputs(source_type, "Use traceId.", version="1")
    projection = project_source_item(
        source_id=f"src-{source_type}", source_type=source_type, run_id="run-title",
        item=item, raw=raw, normalized=normalized,
    )
    kind, *fields = projection.unit_title.text.split("\n")

    assert kind and all(": " in field and field.split(": ", 1)[1] for field in fields)
    assert "None" not in projection.unit_title.text
    assert all(revision.content != projection.unit_title.text for revision in projection.observation_revisions)
    assert not any(observation.provider_key.startswith("$") for observation in projection.observations)


def test_the_stored_projection_carries_the_unit_title_and_one_stored_without_it_names_no_unit():
    projection = jira(issue_type="Defect")
    payload = source_projection_to_payload(projection)

    assert source_projection_from_payload(payload).unit_title == projection.unit_title
    payload.pop("unit_title")
    assert source_projection_from_payload(payload).unit_title is None
    assert "unit_title" not in source_projection_to_payload(replace(projection, unit_title=None))


def test_renaming_a_unit_creates_no_revision():
    first = jira(issue_type="Defect")
    renamed = jira(issue_type="Story", prior=first, run_id="run-2")

    assert renamed.source_unit_revisions[0].id == first.source_unit_revisions[0].id
    assert renamed.deltas[0].axes == frozenset() and not renamed.deltas[0].requires_extraction
    assert renamed.unit_title.text.split("\n")[2] == "Type: Story"


def test_an_unchanged_unit_stored_without_a_title_produces_no_work():
    first = jira()
    again = jira(prior=first, run_id="run-2")
    delta = again.deltas[0]

    assert again.source_unit_revisions[0] == first.source_unit_revisions[0]
    assert (delta.axes, delta.changed_anchors, delta.added_observation_ids, delta.removed_observation_ids) == (
        frozenset(), (), (), (),
    )
    assert not delta.requires_extraction


def test_the_stored_unit_title_is_outside_the_current_representation():
    stored = with_stored_title(jira())

    assert [in_current_representation(revision) for revision in stored.observation_revisions] == [
        False, *(True for _ in jira().observation_revisions),
    ]


@pytest.mark.asyncio
async def test_a_unit_that_stored_its_title_moves_to_a_title_free_revision_and_rebinds_without_a_model_call():
    stored = with_stored_title(jira())
    current = jira(prior=stored, run_id="run-2")
    delta = current.deltas[0]

    assert current.source_unit_revisions[0].id != stored.source_unit_revisions[0].id
    assert {revision.id for revision in current.observation_revisions} == {
        revision.id for revision in stored.observation_revisions if in_current_representation(revision)
    }
    # Only the stored title leaves the Unit; no content changed or was added.
    assert (delta.changed_anchors, delta.added_observation_ids) == ((), ())
    assert delta.removed_observation_ids == ("obs-stored-title",)

    authority = plan_projection_evidence_work(
        current, committed_base_snapshot=committed(stored), reprocess_all_current_observations=False,
    )
    assert isinstance(authority, ExtractionAuthority)
    assert plan_extraction_requests(
        RevisionAssessmentContext(projection=current, base=stored, access_context_hash="scope"),
        authority, extractor=NoopMemoryExtractor(), source_type="jira", doc_type="ticket",
    ).requests == ()

    context = RevisionAssessmentContext(projection=current, base=stored, access_context_hash="scope")
    named = (support_part(current, COMMENT), title_part(stored))
    revision_plan = plan_supports(context, named)
    [support] = revision_plan.supports
    # The part on the stored title is dropped, neither REMOVED nor UNKNOWN.
    assert [part.status for part in support.parts] == [EvidenceCorrespondence.EXACT_UNCHANGED]
    assert support.route is SupportRoute.REBIND_SUPPORT
    assert revision_plan.changes == ()

    client = NoModelClient()
    [result] = (await RevisionWorkExecutor(client=client, model="fixture").assess_many(
        [SupportWorkItem("w0", memory(), named, context)],
    )).values()
    assert result.supported and result.rebound
    assert [part.anchor.observation_id for part in result.memory.resolved_evidence_selection.parts] == [
        support.parts[0].evidence.anchor.observation_id,
    ]


@pytest.mark.asyncio
async def test_partial_coverage_does_not_carry_the_stored_title_and_the_store_retires_it(tmp_path, monkeypatch):
    stored = with_stored_title(jira())
    partial = jira(comments_truncated=True, prior=stored, run_id="run-partial")

    assert partial.coverage is ProjectionCoverage.PARTIAL_PROJECTION
    assert "obsrev-stored-title" not in {revision.id for revision in partial.observation_revisions}
    assert partial.carried_observation_revision_ids == ()
    assert partial.deltas[0].removed_observation_ids == ("obs-stored-title",)

    db = Database(str(tmp_path / "title.db"))
    await db.connect()
    try:
        await db.upsert_source(
            id="src-jira", type="jira", name="Jira", config_json="{}", access_policy="workspace",
            owner_user_id="owner-1",
        )
        declared = database_module.representation_profile_for_observation_contract
        supported_contract = database_module.representation_contract_for_profile
        with monkeypatch.context() as stored_contracts:
            # The contracts under which the title was recorded.
            stored_contracts.setattr(
                database_module, "representation_profile_for_observation_contract",
                lambda *, source_type, observation_type: (
                    LEGACY_TITLE_PROFILE if observation_type == LEGACY_TITLE_TYPE
                    else declared(source_type=source_type, observation_type=observation_type)
                ),
            )
            stored_contracts.setattr(
                database_module, "representation_contract_for_profile",
                lambda profile: object() if profile == LEGACY_TITLE_PROFILE else supported_contract(profile),
            )
            await db.record_source_projection(stored)
        await db.record_source_projection(partial)

        unit_id = partial.source_units[0].id
        current = await db.get_current_source_unit_projection(unit_id)
        assert current.source_unit_revisions[0].id == partial.source_unit_revisions[0].id
        assert "obs-stored-title" not in await db.get_current_source_observation_revisions(unit_id)
    finally:
        await db.close()


def test_a_support_on_a_renamed_unit_is_rebound_with_no_change_to_read():
    first = jira("Payroll run fails")
    renamed = jira("Payroll run fails", prior=first, run_id="run-2")
    context = RevisionAssessmentContext(projection=replace(
        renamed, unit_title=UnitTitle("Jira issue", (("Key", "SFPAY-180000"), ("Summary", "Renamed"))),
    ), base=first, access_context_hash="scope")

    [support] = plan_supports(context, (support_part(first, COMMENT),)).supports

    assert support.route is SupportRoute.REBIND_SUPPORT


def test_the_unit_title_is_never_a_fragment_and_every_extraction_request_shows_it():
    projection = jira(issue_type="Defect")
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
    authority = plan_projection_evidence_work(projection, reprocess_all_current_observations=True)
    requests = plan_extraction_requests(
        context, authority, extractor=NoopMemoryExtractor(), source_type="jira", doc_type="ticket",
    ).requests

    assert requests
    assert all(fragment.presentation_text != projection.unit_title.text for fragment in context.full_fragments)
    for request in requests:
        prompt = MemoryExtractor.projection_fragment_prompt(
            request.catalog, source_type="jira", doc_type="ticket", revision_context=context,
        )
        assert unit_title_block(projection.unit_title) in prompt


@pytest.mark.asyncio
async def test_admission_shows_the_unit_title_it_is_given():
    projection = jira(issue_type="Defect")
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
    catalog = context.catalog(context.full_fragments)
    comment = next(f for f in catalog.fragments if COMMENT in f.presentation_text)
    client = AdmissionClient()

    await admit_candidates([RawMemory(
        content="SFPAY-180000 retains A7 for regular payroll.", memory_type="decision",
        source_observation_id=comment.anchor.observation_id,
        resolved_evidence_selection=catalog.resolve_selection(primary_ref=comment.reference),
    )], client=client, model="fixture", unit_title=projection.unit_title)

    [prompt] = client.prompts
    assert prompt.startswith(unit_title_block(projection.unit_title))
    [request] = client.requests
    # The title is context of the request, never one of its Evidence parts.
    assert all(projection.unit_title.text not in entry["excerpt"] for entry in request["evidence_catalog"].values())


@pytest.mark.asyncio
async def test_the_unit_title_is_shown_in_every_support_step_and_change_impact_request():
    first = jira()
    changed = jira(description="Payroll context changed.", prior=first, run_id="run-2")
    context = RevisionAssessmentContext(projection=changed, base=first, access_context_hash="scope")

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
    block = unit_title_block(changed.unit_title)
    assert all(block in prompt for prompt in (*client.impact_prompts, *client.prompts))


def teams(messages, *, conversation_name="Payroll Dev", prior=None, run_id="run-1"):
    times = [message["time"] for message in messages]
    item = ContentItem(
        item_id="teams-window-1", title=f"Group: {conversation_name} -- {times[0]}-{times[-1]}",
        source_url="https://teams.example.test/conversations/conversation-1", last_modified=NOW, version=run_id,
        extra={"conversation_id": "conversation-1", "window_id": "window-1", "root_message_id": messages[0]["id"],
               "block_start": times[0], "block_end": times[-1]},
    )
    payload = {
        "conversation_id": "conversation-1", "window_id": "window-1", "conversation_type": "group_chat",
        "title": item.title, "channel_name": conversation_name, "conversation_name": conversation_name,
        "messages": messages, "first_message_time": times[0], "last_message_time": times[-1],
    }
    return project_source_item(
        source_id="src-teams", source_type="teams", run_id=run_id, item=item,
        raw=RawContent(item=item, body=json.dumps(payload).encode(), content_type="application/json"),
        normalized=NormalizedContent(item=item, markdown_body="normalized Teams window"),
        prior_unit_revision=prior.source_unit_revisions[0] if prior else None,
        prior_observation_revisions=(
            {revision.observation_id: revision for revision in prior.observation_revisions} if prior else None
        ),
    )


def test_a_teams_window_keeps_its_unit_title_as_it_grows_and_a_renamed_chat_needs_no_revision():
    first_message = {"id": "msg-1", "content": COMMENT, "time": "2026-09-29T02:39:00+00:00"}
    first = teams([first_message])
    grown = teams(
        [first_message, {"id": "msg-2", "content": "Agreed.", "time": "2026-09-29T03:10:00+00:00"}],
        prior=first, run_id="run-2",
    )
    renamed = teams([first_message], conversation_name="Payroll Core", prior=first, run_id="run-3")

    assert first.unit_title.text == (
        "Teams conversation\nConversation type: group_chat\nConversation: Payroll Dev\nFrom: 2026-09-29T02:39:00+00:00"
    )
    assert grown.unit_title == first.unit_title
    assert renamed.source_unit_revisions[0].id == first.source_unit_revisions[0].id
    assert "Conversation: Payroll Core" in renamed.unit_title.text
