"""Catalogs read in source-time order, and record framing is context, never an entry."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole
from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.pipeline.evidence_fragments import build_revision_fragment_index
from memforge.pipeline.memory_extractor import ExtractionReading
from memforge.pipeline.projection_context import (
    CommittedSourceUnitSnapshot,
    plan_projection_evidence_work,
    whole_revision_extraction_authority,
)
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.pipeline.support_reading import EvidenceCorrespondence, correspond_evidence
from tests.test_confluence_native_evidence import projection as confluence_projection

CONVERSATION = (
    ("Ana", "Should the nightly export keep running on the old cluster?", None),
    ("Ben", "I think we should move it this week.", None),
    ("Ana", "Decision: the nightly export moves to the new cluster on Friday.", "msg-1"),
    ("Cat", "Who updates the runbook?", None),
    ("Ben", "I will update the runbook after the move.", "msg-3"),
    ("Ana", "Thanks, closing this thread.", None),
)


def _messages(rows=CONVERSATION) -> list[dict[str, object]]:
    return [
        {"id": f"msg-{index}", "content": text, "time": f"2026-07-14T10:0{index}:00Z", "from": sender,
         **({"reply_to_id": reply} if reply else {})}
        for index, (sender, text, reply) in enumerate(rows)
    ]


def _window(messages, base=None):
    item = ContentItem(
        item_id="window-1", title="Window", source_url="https://example.test/window-1",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc), version="1",
        extra={"conversation_id": "conv-1", "window_id": "window-1", "root_message_id": "msg-0"},
    )
    raw = RawContent(item=item, body=json.dumps({"messages": messages}).encode(), content_type="application/json")
    prior = {} if base is None else {
        "prior_unit_revision": base.source_unit_revisions[0],
        "prior_observation_revisions": {r.observation_id: r for r in base.observation_revisions},
    }
    return project_source_item(
        source_id="src-teams", source_type="teams", run_id=f"run-{len(messages)}", item=item, raw=raw,
        normalized=NormalizedContent(item=item, markdown_body="window"), **prior,
    )


def _reading(projection, base=None) -> ExtractionReading:
    context = RevisionAssessmentContext(projection=projection, base=base, access_context_hash="scope")
    if base is None:
        authority = whole_revision_extraction_authority(projection)
    else:
        authority = plan_projection_evidence_work(
            projection,
            committed_base_snapshot=CommittedSourceUnitSnapshot(base.source_unit_revisions[0], base.observation_revisions),
            reprocess_all_current_observations=False,
        )
    return ExtractionReading.of_authority(context, authority, source_type="teams", doc_type="teams_window")


def _message_ids(projection, fragments) -> list[str]:
    provider_key = {observation.id: observation.provider_key for observation in projection.observations}
    return [provider_key[fragment.anchor.observation_id] for fragment in fragments]


def test_a_conversation_reads_in_the_order_it_was_written() -> None:
    projection = _window(_messages())
    # Revision ids are content hashes, so their order is unrelated to the conversation's.
    assert _message_ids(projection, sorted(
        (f for r in projection.observation_revisions for f in build_revision_fragment_index(r).fragments[:1]),
        key=lambda fragment: fragment.anchor.observation_revision_id,
    )) != [f"msg-{index}" for index in range(len(CONVERSATION))]

    catalog = _reading(projection).catalog

    assert _message_ids(projection, catalog.fragments) == [f"msg-{index}" for index in range(len(CONVERSATION))]
    assert [fragment.reference for fragment in catalog.fragments] == [
        f"p{index:06d}" for index in range(1, len(CONVERSATION) + 1)
    ]


def test_an_observation_without_a_source_time_follows_those_that_have_one() -> None:
    projection = _window(_messages())
    first = next(o.id for o in projection.observations if o.provider_key == "msg-0")
    untimed = replace(projection, observation_revisions=tuple(
        replace(revision, observed_at=None) if revision.observation_id == first else revision
        for revision in projection.observation_revisions
    ))

    catalog = _reading(untimed).catalog

    assert _message_ids(untimed, catalog.fragments) == ["msg-1", "msg-2", "msg-3", "msg-4", "msg-5", "msg-0"]


def test_record_framing_is_shown_on_every_entry_and_never_listed_as_one() -> None:
    projection = _window(_messages())

    reading = _reading(projection)

    assert [fragment.presentation_text.splitlines()[-1] for fragment in reading.catalog.fragments] == [
        text for _sender, text, _reply in CONVERSATION
    ]
    assert all(fragment.primary_eligible for fragment in reading.catalog.fragments)
    reply = reading.catalog.fragments[2].presentation_text
    assert "Reported sender: Ana" in reply
    assert "Message created: 2026-07-14T10:02:00Z" in reply
    assert "Reply referent: msg-1" in reply
    assert len(reading.items) == len(CONVERSATION)


def test_a_contextual_field_outside_the_framing_is_context_and_never_primary() -> None:
    page = confluence_projection("<p>Only after cutoff and approval.</p>")
    context = RevisionAssessmentContext(projection=page, base=None, access_context_hash="scope")

    reading = ExtractionReading.of_authority(
        context, whole_revision_extraction_authority(page), source_type="confluence", doc_type="page",
    )

    title = json.loads(page.observation_revisions[0].content)["title"]
    eligibility = {fragment.presentation_text: fragment.primary_eligible for fragment in reading.catalog.fragments}
    assert eligibility == {"Only after cutoff and approval.": True, title: False}
    assert [group[0].presentation_text for group in reading.items.values()] == ["Only after cutoff and approval."]


def test_changed_framing_changes_every_entry_of_its_record() -> None:
    base = _window(_messages(CONVERSATION[:2]))
    renamed = _messages(CONVERSATION[:2])
    renamed[1]["from"] = "Benjamin"
    target = _window(renamed, base)
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash="scope")

    changed, _removed = context.delta_fragments()

    assert [fragment.presentation_text.splitlines()[-1] for fragment in changed] == [CONVERSATION[1][1]]
    assert "Reported sender: Benjamin" in changed[0].presentation_text


def test_evidence_on_a_framing_field_is_assessed_again_instead_of_reused() -> None:
    projection = _window(_messages(CONVERSATION[:1]))
    revision = projection.observation_revisions[0]
    sender = next(
        fragment for fragment in build_revision_fragment_index(revision).fragments
        if fragment.presentation_text.splitlines()[-1] == "Ana"
    )
    context = RevisionAssessmentContext(
        projection=projection, base=projection, access_context_hash="scope", evidence_revisions=(revision,),
    )

    [correspondence] = correspond_evidence(context, (ActiveSupportEvidence(
        memory_id="memory", source_id=projection.source_id, reference_id="reference", evidence_unit_id="unit",
        role=EvidenceRole.REQUIRED, anchor=sender.anchor, excerpt=sender.presentation_text,
        raw_content_sha256=sender.raw_content_sha256, presentation_sha256=sender.presentation_sha256,
        text_view=sender.text_view,
    ),))

    assert correspondence.status is EvidenceCorrespondence.MODIFIED
