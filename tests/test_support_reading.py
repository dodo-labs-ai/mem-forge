"""Exact prior Evidence correspondence, whole-Support routing, changes and reading order; program only, no model."""

from dataclasses import replace

from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.support_reading import (
    EvidenceCorrespondence as Status,
    SupportRoute,
    SupportWorkItem,
    plan_support_revision,
)
from memforge.source_projection import ProjectionCoverage
from tests.test_revision_assessment import memory, revisions

RULE = "Two reviewers approve US releases."


def part(projection, text, *, role=EvidenceRole.PRIMARY, exact=True, reference_id="e1"):
    """Prior Evidence resolved from an old revision, as the applied Support persisted it."""
    fragment = next(
        f for f in RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope").full_fragments
        if f.presentation_text == text
    )
    return ActiveSupportEvidence(
        memory_id="memory",
        source_id=projection.source_id,
        reference_id=reference_id,
        evidence_unit_id="eu1",
        role=role,
        anchor=fragment.anchor,
        excerpt=text,
        raw_content_sha256=fragment.raw_content_sha256 if exact else None,
        presentation_sha256=fragment.presentation_sha256 if exact else None,
    )


def plan(context, *supports):
    items = [SupportWorkItem(f"w{index}", memory(), support, context) for index, support in enumerate(supports)]
    return plan_support_revision(context, items).supports


def context_for(old, new, **changes):
    base, current = revisions(old, new)
    current = replace(current, **changes) if changes else current
    return base, RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")


def statuses(support):
    return [item.status for item in support.parts]


def test_exact_match_ignores_container_change():
    base, context = context_for(f"# Old heading\n\n{RULE}\n", f"# New heading\n\n{RULE}\n")
    [support] = plan(context, (part(base, RULE),))
    assert statuses(support) == [Status.EXACT_UNCHANGED]
    [current] = support.parts[0].current
    assert current.presentation_text == RULE and current.anchor.observation_revision_id == "rev-primary-v2"
    # The heading changed, so Change Impact decides whether the exact Support is rebound.
    assert support.route is SupportRoute.CHANGE_IMPACT


def test_punctuation_change_is_modified_only_for_its_own_support():
    other = "Hotfixes need one reviewer."
    base, context = context_for(f"{RULE}\n\n{other}\n", f"{RULE[:-1]}!\n\n{other}\n")
    changed, unchanged = plan(context, (part(base, RULE),), (part(base, other),))
    assert statuses(changed) == [Status.MODIFIED] and changed.parts[0].current == ()
    assert statuses(unchanged) == [Status.EXACT_UNCHANGED]
    assert unchanged.matched_refs and not changed.matched_refs


def test_duplicate_exact_text_is_ambiguous():
    base, context = context_for(f"{RULE}\n", f"{RULE}\n\n# Copy\n\n{RULE}\n")
    [support] = plan(context, (part(base, RULE),))
    assert statuses(support) == [Status.AMBIGUOUS] and len(support.parts[0].current) == 2
    assert support.route is SupportRoute.SUPPORT_ASSESSMENT


def test_legacy_part_without_digest_is_never_exact():
    base, _ = revisions(f"{RULE}\n", f"{RULE}\n")
    same_revision = RevisionAssessmentContext(projection=base, base=base, access_context_hash="scope")
    [support] = plan(same_revision, (part(base, RULE, exact=False),))
    assert statuses(support) == [Status.MODIFIED]
    assert support.route is SupportRoute.SUPPORT_ASSESSMENT
    # The Observation revision is unchanged, so the old anchor still locates its current text.
    assert [f.presentation_text for f in support.parts[0].current] == [RULE]


def test_modified_part_is_read_in_the_first_part_even_without_changed_content():
    base, _ = revisions(f"Intro para.\n\n{RULE}\n\nTail para.\n", "")
    same_revision = RevisionAssessmentContext(projection=base, base=base, access_context_hash="scope")
    assert same_revision.delta_fragments() == ((), ())
    items = [SupportWorkItem("w0", memory(), (part(base, RULE, exact=False),), same_revision)]
    revision_plan = plan_support_revision(same_revision, items)
    reading = revision_plan.reading_order(revision_plan.supports)
    assert reading_texts(reading)[0] == [RULE]
    assert reading.first_part_end == {"w0": 1}


def test_modified_part_in_a_changed_revision_still_reads_one_part_first():
    base, context = context_for(f"{RULE}\n\nTail para.\n", f"{RULE}\n\nTail para.\n")
    assert context.delta_fragments() == ((), ())
    items = [SupportWorkItem("w0", memory(), (part(base, RULE, exact=False),), context)]
    revision_plan = plan_support_revision(context, items)
    [support] = revision_plan.supports
    assert statuses(support) == [Status.MODIFIED] and support.parts[0].current == ()
    # Nothing changed and no current text is located, yet the old excerpt still needs one request.
    assert revision_plan.reading_order(revision_plan.supports).first_part_end == {"w0": 1}


def test_other_unit_observation_is_never_rebound():
    base, context = context_for(f"{RULE}\n", f"{RULE}\n")
    elsewhere = replace(
        part(base, RULE),
        anchor=replace(part(base, RULE).anchor, observation_id="obs-other-unit", observation_revision_id="rev-other"),
    )
    [complete] = plan(context, (elsewhere,))
    assert statuses(complete) == [Status.REMOVED] and complete.route is SupportRoute.SUPPORT_ASSESSMENT

    _, partial = context_for(f"{RULE}\n", f"{RULE}\n", coverage=ProjectionCoverage.PARTIAL_PROJECTION)
    [unknown] = plan(partial, (elsewhere,))
    assert statuses(unknown) == [Status.UNKNOWN] and unknown.route is SupportRoute.UNRESOLVED_PARTIAL_COVERAGE


def test_all_exact_without_changed_content_rebinds():
    base, context = context_for(f"{RULE}\n", f"{RULE}\n")
    assert context.delta_fragments() == ((), ())
    items = [SupportWorkItem("w0", memory(), (part(base, RULE), part(base, "Country: US.", role=EvidenceRole.REQUIRED)), context)]
    revision_plan = plan_support_revision(context, items)
    [support] = revision_plan.supports
    assert support.route is SupportRoute.REBIND_SUPPORT
    primary, required = support.matched_refs
    selection = revision_plan.catalog.resolve_selection(primary_ref=primary, required_refs=(required,))
    assert [p.anchor.observation_revision_id for p in selection.parts] == ["rev-primary-v2", "rev-context"]


def test_all_exact_with_changed_content_routes_to_change_impact():
    base, context = context_for(f"{RULE}\n", f"{RULE}\n\nNew unrelated note.\n")
    [support] = plan(context, (part(base, RULE),))
    assert statuses(support) == [Status.EXACT_UNCHANGED]
    assert support.route is SupportRoute.CHANGE_IMPACT


def test_removed_content_counts_as_changed():
    base, context = context_for(f"{RULE}\n\nOld distant exception.\n", f"{RULE}\n")
    changed, removed = context.delta_fragments()
    assert changed == () and [fragment.presentation_text for fragment in removed] == ["Old distant exception."]
    [support] = plan(context, (part(base, RULE),))
    assert statuses(support) == [Status.EXACT_UNCHANGED]
    assert support.route is SupportRoute.CHANGE_IMPACT


def test_unknown_part_takes_priority_over_modified():
    base, current = revisions(f"{RULE}\n", "One reviewer approves US releases.\n")
    current = replace(
        current,
        observations=current.observations[:1],
        coverage=ProjectionCoverage.PARTIAL_PROJECTION,
        carried_observation_revision_ids=("rev-context",),
        deltas=(replace(current.deltas[0], coverage=ProjectionCoverage.PARTIAL_PROJECTION, added_observation_ids=()),),
    )
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    [support] = plan(context, (part(base, RULE), part(base, "Country: US.", role=EvidenceRole.REQUIRED)))
    # The carried context Observation was not returned, even though its revision is unchanged.
    assert statuses(support) == [Status.MODIFIED, Status.UNKNOWN]
    assert support.route is SupportRoute.UNRESOLVED_PARTIAL_COVERAGE
    assert support.unknown_observation_ids == ("obs-context",)


def test_without_usable_baseline_exact_support_is_assessed_not_rebound():
    base, current = revisions(f"{RULE}\n", f"{RULE}\n")
    no_baseline = RevisionAssessmentContext(projection=current, base=None, access_context_hash="scope")
    [support] = plan(no_baseline, (part(base, RULE),))
    assert statuses(support) == [Status.EXACT_UNCHANGED]
    assert support.route is SupportRoute.SUPPORT_ASSESSMENT


def test_absent_member_observation_is_removed_under_complete_coverage():
    base, current = revisions(f"{RULE}\n", f"{RULE}\n")
    unit = replace(current.source_unit_revisions[0], observation_revision_ids=("rev-primary-v2",))
    current = replace(
        current,
        observations=current.observations[:1],
        observation_revisions=current.observation_revisions[:1],
        source_unit_revisions=(unit,),
        deltas=(replace(current.deltas[0], added_observation_ids=("obs-primary",)),),
    )
    context = RevisionAssessmentContext(projection=current, base=base, access_context_hash="scope")
    [support] = plan(context, (part(base, RULE), part(base, "Country: US.", role=EvidenceRole.REQUIRED)))
    assert statuses(support) == [Status.EXACT_UNCHANGED, Status.REMOVED]
    assert support.route is SupportRoute.SUPPORT_ASSESSMENT


def reading_texts(reading):
    return [
        [f.presentation_text for f in part.fragments] or [part.removed["text"]]
        for part in reading.parts
    ]


def test_reading_order_puts_changed_removed_and_own_evidence_first():
    other = "Other rule."
    old = f"Intro para.\n\n{RULE}\n\n{other}\n\nDeleted note.\n\n- item a\n- item b\n\nTail para.\n"
    new = f"Intro para.\n\n{RULE}\n\n{other}\n\n- item a\n- item b\n- item c\n\nTail para.\n\nAdded note.\n"
    base, context = context_for(old, new)
    items = [
        SupportWorkItem("w0", memory(), (part(base, RULE),), context),
        SupportWorkItem("w1", memory(), (part(base, other),), context),
    ]
    revision_plan = plan_support_revision(context, items)
    reading = revision_plan.reading_order(revision_plan.supports)
    assert reading_texts(reading) == [
        # Changed current groups: the whole changed list is one group.
        ["- item a", "- item b", "- item c"], ["Added note."],
        # Removed old text.
        ["Deleted note."],
        # Each Support's own exactly matched prior Evidence.
        [RULE], [other],
        # Everything else, in document order.
        ["Country: US."], ["Intro para."], ["Tail para."],
    ]
    assert reading.parts[2].label == "removed:obs-primary:rev-primary:h000000"
    assert [entry["text"] for entry in reading.removed] == ["Deleted note."]
    # Each Support's first part ends after its own Evidence group.
    assert reading.first_part_end == {"w0": 4, "w1": 5}


def test_without_usable_baseline_first_part_is_the_whole_revision():
    base, current = revisions(f"Intro para.\n\n{RULE}\n", f"Intro para.\n\n{RULE}\n\nNew note.\n")
    no_baseline = RevisionAssessmentContext(projection=current, base=None, access_context_hash="scope")
    items = [SupportWorkItem("w0", memory(), (part(base, RULE),), no_baseline)]
    revision_plan = plan_support_revision(no_baseline, items)
    assert revision_plan.changes == ()
    reading = revision_plan.reading_order(revision_plan.supports)
    assert reading_texts(reading) == [["Country: US."], ["Intro para."], [RULE], ["New note."]]
    assert reading.first_part_end == {"w0": len(reading.parts)}


def test_changes_hold_changed_groups_then_removed_text():
    old = f"Intro para.\n\n{RULE}\n\nDeleted note.\n\n- item a\n- item b\n\nTail para.\n"
    new = f"Intro para.\n\n{RULE}\n\n- item a\n- item b\n- item c\n\nTail para.\n\nAdded note.\n"
    base, context = context_for(old, new)
    revision_plan = plan_support_revision(context, [SupportWorkItem("w0", memory(), (part(base, RULE),), context)])
    # The whole changed list is one group, in document order; removed old text follows.
    assert [
        [f.presentation_text for f in change.fragments] or [change.removed["text"]] for change in revision_plan.changes
    ] == [["- item a", "- item b", "- item c"], ["Added note."], ["Deleted note."]]
    by_ref = {f.reference: f.presentation_text for f in revision_plan.catalog.fragments}
    # Only the added list item is changed; its unchanged siblings are context.
    assert sorted(by_ref[ref] for ref in revision_plan.changed_refs) == ["- item c", "Added note."]


def test_unchanged_revision_has_no_changes():
    base, context = context_for(f"{RULE}\n", f"{RULE}\n")
    revision_plan = plan_support_revision(context, [SupportWorkItem("w0", memory(), (part(base, RULE),), context)])
    assert revision_plan.changes == () and revision_plan.changed_refs == frozenset()
    assert revision_plan.supports[0].route is SupportRoute.REBIND_SUPPORT


def test_reading_order_is_built_for_assessed_supports_only():
    other = "Other rule."
    old = f"{RULE}\n\nIntro para.\n\n{other}\n\nTail para.\n"
    new = f"{RULE[:-1]}!\n\nIntro para.\n\n{other}\n\nTail para.\n\nAdded note.\n"
    base, context = context_for(old, new)
    items = [
        SupportWorkItem("modified", memory(), (part(base, RULE),), context),
        SupportWorkItem("exact", memory(), (part(base, other),), context),
    ]
    revision_plan = plan_support_revision(context, items)
    modified, exact = revision_plan.supports
    assert modified.route is SupportRoute.SUPPORT_ASSESSMENT and exact.route is SupportRoute.CHANGE_IMPACT
    reading = revision_plan.reading_order([modified])
    assert reading.parts[:len(revision_plan.changes)] == revision_plan.changes
    # The exact Support's own group is not pulled forward for a cohort that does not read it.
    first = reading_texts(reading)[:reading.first_part_end["modified"]]
    assert [other] not in first and reading.first_part_end == {"modified": len(revision_plan.changes)}


def test_ambiguous_candidates_are_all_in_the_first_part():
    old = f"{RULE}\n\nIntro para.\n\nTail para.\n"
    new = f"{RULE}\n\nIntro para.\n\n{RULE}\n\nTail para.\n"
    base, context = context_for(old, new)
    items = [SupportWorkItem("w0", memory(), (part(base, RULE),), context)]
    revision_plan = plan_support_revision(context, items)
    reading = revision_plan.reading_order(revision_plan.supports)
    first = reading_texts(reading)[:reading.first_part_end["w0"]]
    assert first.count([RULE]) == 2 and ["Tail para."] not in first
