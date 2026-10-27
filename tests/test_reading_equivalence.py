"""One reading relation decides Primary authority, the Support delta and exact correspondence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole
from memforge.pipeline.evidence_fragments import build_revision_fragment_index, canonical_record_is_tombstoned
from memforge.pipeline.projection_context import (
    CommittedSourceUnitSnapshot,
    ExtractionAuthority,
    ProjectionEvidencePlanningFailure,
    ProjectionEvidencePlanningFailureCode,
    plan_projection_evidence_work,
)
from memforge.pipeline.reading_equivalence import compare_readings
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.support_reading import (
    EvidenceCorrespondence,
    SupportRoute,
    SupportWorkItem,
    plan_support_revision,
)
from memforge.source_adapters.contracts import (
    CanonicalRecordField,
    CanonicalRecordSchema,
    EvidenceRepresentationContract,
    canonical_record_profile,
)
from memforge.source_projection import AnchorKind, SourceAnchor, SourceObservationRevision
from memforge.source_representation import PLAIN_TEXT_PROFILE, _SUPPORTED_REPRESENTATION_CONTRACTS
from tests.test_confluence_native_evidence import projection as page
from tests.test_reading_order_and_record_framing import CONVERSATION, _messages, _window
from tests.test_revision_assessment import memory, revisions


def _plan(base, target, committed=None):
    return plan_projection_evidence_work(
        target,
        committed_base_snapshot=committed or CommittedSourceUnitSnapshot(
            base.source_unit_revisions[0], base.observation_revisions,
        ),
        reprocess_all_current_observations=False,
    )


def _authorized(base, target) -> list[str]:
    """The current entries Claim Extraction may cite as Primary."""
    authority = _plan(base, target)
    assert isinstance(authority, ExtractionAuthority)
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash="scope")
    return [
        fragment.presentation_text for fragment in context.full_fragments
        if fragment.primary_eligible and authority.authorizes(fragment)
    ]


def _support(base, target, text):
    """The Support plan of one claim whose Primary is the base Fragment that shows ``text``."""
    old = next(
        fragment for revision in base.observation_revisions
        for fragment in build_revision_fragment_index(revision).fragments
        if fragment.presentation_text.endswith(text)
    )
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash="scope")
    part = ActiveSupportEvidence(
        memory_id="memory", source_id=base.source_id, reference_id="e", evidence_unit_id="eu",
        role=EvidenceRole.PRIMARY, anchor=old.anchor, excerpt=old.presentation_text,
        raw_content_sha256=old.raw_content_sha256, presentation_sha256=old.presentation_sha256,
        text_view=old.text_view,
    )
    return context, plan_support_revision(context, [SupportWorkItem("w", memory(), (part,), context)]).supports[0]


def _spanning(rows: int) -> str:
    return (
        '<table><tr><th>Group</th><th>Rule</th></tr>'
        f'<tr><td rowspan="{rows}">Payroll</td><td>Rule A</td></tr>'
        + "".join(f"<tr><td>Rule {chr(66 + index)}</td></tr>" for index in range(rows - 1))
        + "</table>"
    )


def _drawio(parameters: str) -> str:
    return (
        '<ac:structured-macro ac:name="drawio"><ac:parameter ac:name="diagramName">flow</ac:parameter>'
        f"{parameters}</ac:structured-macro><p>Approval required.</p>"
    )


@pytest.mark.parametrize("old, new", [
    ('<p>Approval required.&nbsp;</p><p>Other.</p>', '<p>Approval required. </p><p>Other.</p>'),
    ('<p>&nbsp;Approval required.</p>', '<p>Approval required.</p>'),
    (_drawio(""), _drawio('<ac:parameter ac:name="baseUrl"></ac:parameter>')),
    ('<pre>keep A7</pre><p>Approval required.</p>', '<pre style="color: red">keep A7</pre><p>Approval required.</p>'),
], ids=["trailing-nbsp", "leading-nbsp", "empty-macro-parameter", "pre-style"])
def test_a_storage_edit_the_reading_never_shows_authorizes_and_changes_nothing(old, new) -> None:
    base = page(old)
    target = page(new, base)
    assert target.observation_revisions[0].id != base.observation_revisions[0].id

    assert _authorized(base, target) == []
    context, support = _support(base, target, "Approval required.")
    assert context.delta_fragments() == ((), ())
    assert support.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert support.route is SupportRoute.REBIND_SUPPORT


def test_rows_added_under_a_spanning_cell_authorize_only_the_new_row() -> None:
    base = page(_spanning(2))
    target = page(_spanning(3), base)

    assert _authorized(base, target) == ["Group: Payroll [cell from row 2]\nRule: Rule C"]
    context, support = _support(base, target, "Rule: Rule B")
    changed, removed = context.delta_fragments()
    assert [fragment.presentation_text for fragment in changed] == ["Group: Payroll [cell from row 2]\nRule: Rule C"]
    assert removed == ()
    # The covered row reads as it did; the new row is read with it as the revision's change.
    assert support.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert support.route is SupportRoute.CHANGE_IMPACT


@pytest.mark.parametrize("old, new, governed", [
    ('<h1>Rules</h1><p>A7 applies.</p><h1>Other</h1><p>B8 applies.</p>',
     '<h1>Policy</h1><p>A7 applies.</p><h1>Other</h1><p>B8 applies.</p>', ["Policy", "A7 applies."]),
    ('<table><tr><th>Rule</th></tr><tr><td>A7</td></tr></table><p>B8 applies.</p>',
     '<table><tr><th>Instruction</th></tr><tr><td>A7</td></tr></table><p>B8 applies.</p>',
     ["Instruction: Instruction", "Instruction: A7"]),
    ('<p>Only when approved:</p><ul><li>A7</li><li>B8</li></ul><p>Tail.</p>',
     '<p>Only when rejected:</p><ul><li>A7</li><li>B8</li></ul><p>Tail.</p>', ["Only when rejected:", "- A7\n- B8"]),
], ids=["heading", "table-header", "list-lead-in"])
def test_a_changed_governing_structure_authorizes_what_it_governs(old, new, governed) -> None:
    base = page(old)
    target = page(new, base)

    assert _authorized(base, target) == governed


def test_changed_governing_framing_authorizes_every_entry_it_frames() -> None:
    base = _window(_messages(CONVERSATION[:2]))
    renamed = _messages(CONVERSATION[:2])
    renamed[1]["from"] = "Benjamin"
    target = _window(renamed, base)

    [authorized] = _authorized(base, target)
    assert authorized.endswith(CONVERSATION[1][1]) and "Reported sender: Benjamin" in authorized
    context, support = _support(base, target, CONVERSATION[1][1])
    # The message's own reading is unchanged, so its Support is judged against the change.
    assert support.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert support.route is SupportRoute.CHANGE_IMPACT
    assert [fragment.presentation_text for fragment in context.delta_fragments()[0]] == [authorized]


def test_a_restored_record_authorizes_what_it_shows_again() -> None:
    deleted = _messages(CONVERSATION[:1])
    deleted[0]["deletedDateTime"] = "2026-07-14T11:00:00Z"
    base = _window(deleted)
    target = _window(_messages(CONVERSATION[:1]), base)

    [authorized] = _authorized(base, target)
    assert authorized.endswith(CONVERSATION[0][1])


def _edited_window():
    base = _window(_messages(CONVERSATION[:1]))
    edited = _messages(CONVERSATION[:1])
    edited[0]["content"] = "Should the nightly export move to the new cluster?"
    return base, _window(edited, base)


def _committed_with_base_profile(base, profile, *, content=None):
    [revision] = base.observation_revisions
    pinned = replace(revision, evidence_profile=profile, content=content if content is not None else revision.content)
    return CommittedSourceUnitSnapshot(base.source_unit_revisions[0], (pinned,))


def test_a_base_unreadable_under_its_own_older_profile_leaves_the_whole_observation_to_read() -> None:
    base, target = _edited_window()
    [current] = target.observation_revisions
    unregistered = replace(current.evidence_profile, schema_version=99)

    authority = _plan(base, target, _committed_with_base_profile(base, unregistered))

    assert isinstance(authority, ExtractionAuthority)
    assert authority.ranges_by_observation_id == {current.observation_id: None}


def test_a_base_that_cannot_compile_under_the_current_profile_is_a_planning_failure() -> None:
    base, target = _edited_window()
    [current] = target.observation_revisions

    failure = _plan(base, target, _committed_with_base_profile(base, current.evidence_profile, content="{"))

    assert isinstance(failure, ProjectionEvidencePlanningFailure)
    assert failure.code is ProjectionEvidencePlanningFailureCode.CANONICAL_FIELD_MAPPING_INVALID


def test_a_deleted_record_authorizes_nothing_without_reading_its_base() -> None:
    base = _window(_messages(CONVERSATION[:1]))
    deleted = _messages(CONVERSATION[:1])
    deleted[0]["deleted_at"] = "2026-07-14T11:00:00Z"
    target = _window(deleted, base)
    [current] = target.observation_revisions
    assert canonical_record_is_tombstoned(current)

    # A same-profile base that cannot be compiled does not matter: a deleted record reads nothing.
    authority = _plan(base, target, _committed_with_base_profile(
        base, current.evidence_profile, content=base.observation_revisions[0].content[:-1],
    ))

    assert authority == ExtractionAuthority({})


def _document(base_content, target_content, *, base_profile=None):
    """A Markdown document body edited between a committed base and the current revision."""
    base, target = revisions(base_content, target_content)
    if base_profile is not None:
        pinned = replace(base.observation_revisions[0], evidence_profile=base_profile)
        base = replace(base, observation_revisions=(pinned, *base.observation_revisions[1:]))
    current = target.observation_revisions[0]
    changed = SourceAnchor(
        kind=AnchorKind.REVISION_RANGE, observation_id=current.observation_id,
        observation_revision_id=current.id, range_start=0, range_end=len(current.content),
    )
    target = replace(target, deltas=(replace(target.deltas[0], added_observation_ids=(), changed_anchors=(changed,)),))
    return base, target


def test_authority_and_the_support_delta_read_the_same_changes_in_a_document() -> None:
    base, target = _document(
        "# Rules\n\n- Approve A7.\n- Audit B8.\n\nKeep every log.\n",
        "# Rules\n\n- Approve A9.\n- Audit B8.\n\nKeep every log.\n",
    )

    authorized = _authorized(base, target)
    changed, removed = RevisionAssessmentContext(projection=target, base=base, access_context_hash="scope").delta_fragments()

    assert authorized == ["- Approve A9."]
    assert [fragment.presentation_text for fragment in changed] == authorized
    assert [fragment.presentation_text for fragment in removed] == ["- Approve A7."]


def test_a_document_compared_with_a_base_of_another_profile_is_read_again_whole() -> None:
    base, target = _document("Use approval before release.\n", "# Release\n\nUse approval before release.\n",
                             base_profile=PLAIN_TEXT_PROFILE)
    [primary, _context] = target.observation_revisions

    authority = _plan(base, target)
    changed, removed = RevisionAssessmentContext(projection=target, base=base, access_context_hash="scope").delta_fragments()

    assert isinstance(authority, ExtractionAuthority)
    assert authority.ranges_by_observation_id == {primary.observation_id: None}
    assert [fragment.presentation_text for fragment in changed] == ["# Release", "Use approval before release."]
    assert [fragment.presentation_text for fragment in removed] == ["Use approval before release."]


_RECORD = {"body": "Approval is required.\n\nAudit every run.", "owner": "team_alpha", "note": "draft"}


def _schema(version: int, **changes) -> CanonicalRecordSchema:
    return replace(CanonicalRecordSchema(
        name="test-record", version=version,
        fields=(CanonicalRecordField("/body", nested_profile="plain-text"),
                CanonicalRecordField("/owner", contextual=True)),
        model_interpretation="The owner decides the rule.",
    ), **changes)


def _readings(monkeypatch, base_schema, target_schema, base_record=_RECORD, target_record=_RECORD):
    revisions = []
    for name, schema, record in (("base", base_schema, base_record), ("target", target_schema, target_record)):
        profile = canonical_record_profile(schema.name, schema.version)
        monkeypatch.setitem(_SUPPORTED_REPRESENTATION_CONTRACTS, profile, EvidenceRepresentationContract(profile, schema))
        revisions.append(SourceObservationRevision(
            id=f"rev-{name}", observation_id="obs", semantic_hash=name,
            content=json.dumps(record, sort_keys=True), evidence_profile=profile,
        ))
    base, target = revisions
    fragments = {
        fragment.anchor: fragment.presentation_text for fragment in build_revision_fragment_index(target).fragments
    }
    return [fragments[anchor] for anchor in compare_readings(base, target).changed()]


def test_an_upgrade_that_reads_the_same_changes_nothing(monkeypatch) -> None:
    assert _readings(monkeypatch, _schema(1), _schema(2)) == []


def test_a_field_promoted_to_selectable_is_changed(monkeypatch) -> None:
    promoted = _schema(2, fields=(CanonicalRecordField("/body", nested_profile="plain-text"),
                                  CanonicalRecordField("/owner")))

    # The body is no longer read with the owner as its context, so it changed too.
    assert _readings(monkeypatch, _schema(1), promoted) == ["Approval is required.", "Audit every run.", "team_alpha"]


def test_a_changed_model_interpretation_changes_every_fragment(monkeypatch) -> None:
    reinterpreted = _schema(2, model_interpretation="The owner only proposes the rule.")

    assert _readings(monkeypatch, _schema(1), reinterpreted) == ["Approval is required.", "Audit every run.", "team_alpha"]


def test_a_changed_descriptor_label_changes_what_it_labels(monkeypatch) -> None:
    labelled = _schema(2, fields=(CanonicalRecordField("/body", nested_profile="plain-text", label="Decision"),
                                  CanonicalRecordField("/owner", contextual=True)))

    assert _readings(monkeypatch, _schema(1), labelled) == [
        "Decision:\nApproval is required.", "Decision:\nAudit every run.",
    ]


def test_framing_the_target_does_not_declare_governs(monkeypatch) -> None:
    framed = _schema(1, framing_fields=(("/note", "Note"),))

    changed = _readings(monkeypatch, framed, framed, target_record={**_RECORD, "note": "final"})

    assert changed == ["Note: final\nApproval is required.", "Note: final\nAudit every run.", "Note: final\nteam_alpha"]


def test_only_a_contextual_field_can_be_shown_without_governing_interpretation() -> None:
    with pytest.raises(ValueError, match="contextual"):
        CanonicalRecordField("/body", governs_interpretation=False)
