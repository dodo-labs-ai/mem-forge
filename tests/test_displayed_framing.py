"""Edit times are shown with comments and messages but never change what they say."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.pipeline.evidence_fragments import build_revision_fragment_index
from memforge.pipeline.revision_work import RevisionWorkExecutor
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.pipeline.support_reading import EvidenceCorrespondence, SupportRoute
from memforge.source_adapters.contracts import canonical_record_profile
from memforge.source_projection import SourceObservationRevision
from memforge.source_representation import representation_contract_for_profile
from tests.test_reading_equivalence import _authorized, _plan, _support
from tests.test_reading_order_and_record_framing import CONVERSATION, _messages, _window

_ITEM = ContentItem(
    item_id="jira-PAYR-123", title="Payroll", source_url="https://jira.example.test/browse/PAYR-123",
    last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc), version="1", extra={"issue_key": "PAYR-123"},
)


def _comment(paragraphs, updated):
    return {
        "id": "500", "body": "\n\n".join(paragraphs),
        "renderedBody": "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs),
        "author": {"displayName": "Ana", "key": "ana"},
        "created": "2026-07-01T10:00:00.000+0000", "updated": updated,
    }


def _issue(*comments, prior=None):
    payload = {
        "id": "10012", "key": "PAYR-123",
        "fields": {"summary": "Payroll", "description": "A7 processing context", "status": None, "priority": None,
                   "assignee": None, "labels": [], "resolution": None, "updated": "2026-07-15T00:00:00Z"},
        "renderedFields": {"description": "<p>A7 processing context</p>"},
        "_comments": list(comments), "_comments_included": True, "_comments_total": len(comments),
        "changelog": {"startAt": 0, "histories": [], "total": 0},
    }
    return project_source_item(
        source_id="src-j", source_type="jira", run_id="run-2" if prior else "run-1", item=_ITEM,
        raw=RawContent(item=_ITEM, body=json.dumps(payload).encode(), content_type="application/json"),
        normalized=NormalizedContent(item=_ITEM, markdown_body="normalized Jira"),
        prior_unit_revision=prior.source_unit_revisions[0] if prior else None,
        prior_observation_revisions={r.observation_id: r for r in prior.observation_revisions} if prior else None,
    )


def test_a_comment_body_edit_authorizes_only_the_edited_structure() -> None:
    base = _issue(_comment(["Retain A7 for the payroll run.", "Cutoff is Friday."], "2026-07-01T10:00:00.000+0000"))
    target = _issue(
        _comment(["Retain A7 for the payroll run.", "Cutoff is Thursday."], "2026-07-02T09:00:00.000+0000"),
        prior=base,
    )

    [authorized] = _authorized(base, target)
    assert authorized.endswith("Cutoff is Thursday.")
    assert "Comment updated: 2026-07-02T09:00:00.000+0000" in authorized

    context, unchanged = _support(base, target, "Retain A7 for the payroll run.")
    changed, removed = context.delta_fragments()
    assert [fragment.presentation_text.splitlines()[-1] for fragment in changed] == ["Cutoff is Thursday."]
    assert [fragment.presentation_text.splitlines()[-1] for fragment in removed] == ["Cutoff is Friday."]
    assert unchanged.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert unchanged.route is SupportRoute.CHANGE_IMPACT
    _, edited = _support(base, target, "Cutoff is Friday.")
    assert edited.parts[0].status is EvidenceCorrespondence.MODIFIED


def test_an_edit_time_alone_authorizes_nothing_and_changes_no_entry() -> None:
    base = _window(_messages(CONVERSATION[:2]))
    edited = _messages(CONVERSATION[:2])
    edited[1]["edited_time"] = "2026-07-14T10:05:00Z"
    target = _window(edited, base)
    assert {r.id for r in target.observation_revisions} != {r.id for r in base.observation_revisions}

    assert _authorized(base, target) == []
    context, support = _support(base, target, CONVERSATION[1][1])
    assert context.delta_fragments() == ((), ())
    assert support.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert support.route is SupportRoute.REBIND_SUPPORT
    # The rebound Evidence shows the edit time it is now displayed with.
    rebound = RevisionWorkExecutor(client=None, model="unused")._rebound(
        context.catalog(context.full_fragments), support,
    )
    assert "Message edited: 2026-07-14T10:05:00Z" in rebound.memory.resolved_evidence_selection.parts[0].excerpt


def _as_written_by(projection, schema_name: str, version: int):
    """The projection as an earlier version of one record schema wrote the same records."""
    revisions = []
    for revision in projection.observation_revisions:
        profile = revision.evidence_profile
        if profile is not None and profile.schema_name == schema_name:
            current = f'"{schema_name}:{profile.schema_version}"'
            assert current in revision.content
            revision = replace(
                revision, id=f"{revision.id}-v{version}", semantic_hash=f"{revision.semantic_hash}-v{version}",
                content=revision.content.replace(current, f'"{schema_name}:{version}"'),
                evidence_profile=replace(profile, schema_version=version),
            )
        revisions.append(revision)
    renamed = {old.id: new.id for old, new in zip(projection.observation_revisions, revisions, strict=True)}
    unit = replace(projection.source_unit_revisions[0],
                   observation_revision_ids=tuple(revision.id for revision in revisions))
    deltas = tuple(
        replace(delta, changed_anchors=tuple(
            replace(anchor, observation_revision_id=renamed[anchor.observation_revision_id])
            for anchor in delta.changed_anchors
        ))
        for delta in projection.deltas
    )
    return replace(projection, observation_revisions=tuple(revisions), source_unit_revisions=(unit,), deltas=deltas)


_COMMENT = _comment(["Retain A7 for the payroll run.", "Cutoff is Friday."], "2026-07-02T09:00:00.000+0000")


def _edited_messages():
    messages = _messages(CONVERSATION[:2])
    messages[1]["edited_time"] = "2026-07-14T10:05:00Z"
    return messages


@pytest.mark.parametrize("schema_name, earlier, current, text", [
    ("jira-comment", 5, lambda prior: _issue(_COMMENT, prior=prior), "Cutoff is Friday."),
    ("teams-message", 2, lambda prior: _window(_edited_messages(), prior), CONVERSATION[1][1]),
])
def test_an_upgrade_that_only_declares_the_edit_time_displayed_reads_the_same(
    schema_name, earlier, current, text,
) -> None:
    base = _as_written_by(current(None), schema_name, earlier)
    target = current(base)
    changed = [
        revision for revision in target.observation_revisions
        if revision.evidence_profile.schema_name == schema_name
    ]
    assert changed and all(revision.id not in {r.id for r in base.observation_revisions} for revision in changed)

    authority = _plan(base, target)
    assert all(revision.observation_id not in authority.ranges_by_observation_id for revision in changed)
    context, support = _support(base, target, text)
    assert context.delta_fragments() == ((), ())
    assert support.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert support.route is SupportRoute.REBIND_SUPPORT


@pytest.mark.parametrize("schema_name, earlier, current, record", [
    ("jira-comment", 5, 6, {
        "body": "<p>Retain A7 for the payroll run.</p><p>Cutoff is Friday.</p>", "native_body": "Retain A7.",
        "issue_key": "PAYR-123", "author": {"displayName": "Ana", "key": "ana"},
        "created": "2026-07-01T10:00:00.000+0000", "updated": "2026-07-02T09:00:00.000+0000", "attachments": [],
    }),
    ("teams-message", 2, 3, {
        "content": "Decision: keep A7.\n\nRollout starts Monday.", "attachments": [], "deleted": None,
        "from": "Ana", "time": "2026-07-14T10:00:00Z", "edited_time": "2026-07-14T10:05:00Z",
        "reply_to_id": "msg-0", "conversation_id": "conv-1", "conversation_name": "team_alpha",
    }),
])
def test_declaring_the_edit_time_displayed_persists_exactly_what_the_earlier_version_did(
    schema_name, earlier, current, record,
) -> None:
    def persisted(version):
        content = json.dumps({**record, "representation": f"{schema_name}:{version}"}, sort_keys=True)
        profile = canonical_record_profile(schema_name, version)
        assert representation_contract_for_profile(profile) is not None
        index = build_revision_fragment_index(SourceObservationRevision(
            id="rev", observation_id="obs", semantic_hash="hash", content=content, evidence_profile=profile,
        ))
        assert index.complete and index.fragments
        return [
            (f.anchor, f.fragment_type, f.primary_eligible, f.raw_content_sha256, f.presentation_text,
             f.presentation_sha256, f.content_value, f.interpretation_origins, f.text_view)
            for f in index.fragments
        ]

    assert persisted(current) == persisted(earlier)
