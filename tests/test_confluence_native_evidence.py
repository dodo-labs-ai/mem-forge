"""Provider-owned native compilation through the actual projection/reading seam."""

import hashlib
import json
from dataclasses import replace

import pytest

from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole
from memforge.pipeline.evidence_fragments import build_revision_fragment_index
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.projection_fragments import SupportRevalidationLimitation
from memforge.pipeline.support_reading import EvidenceCorrespondence, SupportRoute, SupportWorkItem, plan_support_revision
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_adapters.confluence_storage import parse_storage
from memforge.source_representation import MARKDOWN_STRUCTURAL_PROFILE
from memforge.source_projection import ProjectionCoverage
from tests.test_revision_assessment import memory
from tests.test_source_projection_adapters import _inputs, _item


def projection(storage, prior=None, normalized="INTENTIONALLY WRONG NORMALIZATION"):
    item = _item(item_id="confluence-123", title="Rules", extra={"page_id": "123"})
    raw, body = _inputs(item, storage.encode(), markdown=normalized)
    return project_source_item(
        source_id="src", source_type="confluence", run_id="r2" if prior else "r1",
        item=item, raw=raw, normalized=body,
        prior_unit_revision=prior.source_unit_revisions[0] if prior else None,
        prior_observation_revisions={r.observation_id: r for r in prior.observation_revisions} if prior else None,
    )


def support(base, fragment):
    return ActiveSupportEvidence(
        memory_id="memory", source_id=base.source_id, reference_id="e", evidence_unit_id="eu",
        role=EvidenceRole.PRIMARY, anchor=fragment.anchor, excerpt=fragment.presentation_text,
        raw_content_sha256=fragment.raw_content_sha256, presentation_sha256=fragment.presentation_sha256,
    )


def planned(base, target, fragment):
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash="scope")
    result = plan_support_revision(context, [SupportWorkItem("w", memory(), (support(base, fragment),), context)])
    return context, result.supports[0]


def test_native_source_is_pinned_instead_of_lossy_normalized_body():
    native = '<h2>Eligibility</h2><p>Only after cutoff &amp; approval.</p>'
    projected = projection(native)
    revision = projected.observation_revisions[0]
    assert json.loads(revision.content)["body"] == native
    index = build_revision_fragment_index(revision)
    assert not index.errors
    assert [f.presentation_text for f in index.fragments] == ["Eligibility", "Only after cutoff & approval.", "Rules"]
    assert all("WRONG NORMALIZATION" not in f.presentation_text for f in index.fragments)
    for fragment in index.fragments:
        anchor = fragment.anchor
        assert hashlib.sha256(revision.content[anchor.range_start:anchor.range_end].encode()).hexdigest() == fragment.raw_content_sha256


def test_table_row_status_and_issue_are_readable_without_inferred_outcome():
    native = '<table><tr><th>Case</th><th>Expected behavior</th><th>Result</th></tr><tr><td>14</td><td>Late hire is rejected only for the current period.</td><td><ac:structured-macro ac:name="status" ac:macro-id="a"><ac:parameter ac:name="title">Failed</ac:parameter><ac:parameter ac:name="colour">Red</ac:parameter></ac:structured-macro><ac:structured-macro ac:name="jira"><ac:parameter ac:name="key">PAY-42</ac:parameter><ac:parameter ac:name="server">Payroll</ac:parameter></ac:structured-macro></td></tr></table>'
    base = projection(native)
    context = RevisionAssessmentContext(projection=base, base=None, access_context_hash="scope")
    row = next(f for f in context.full_fragments if f.fragment_type.endswith("table-row"))
    assert "Case: 14" in row.presentation_text
    assert "only for the current period" in row.presentation_text
    assert "Status: Failed (colour: Red)" in row.presentation_text
    assert "Issue: PAY-42 (server: Payroll)" in row.presentation_text
    assert json.loads(base.observation_revisions[0].content)["body"] == native
    assert "<ac:" not in row.presentation_text and "<tr>" not in row.presentation_text
    context_anchors = context.reading_context((row,))
    assert any(f.fragment_type.endswith("table-header") and f.anchor in context_anchors for f in context.full_fragments)


def test_native_code_cdata_is_literal_while_formatted_children_are_decoded():
    parsed = parse_storage('<p><code><strong>A7</strong> &amp; B</code></p><ac:structured-macro ac:name="code"><ac:parameter ac:name="language">xml</ac:parameter><ac:plain-text-body><![CDATA[<rule>&amp;</rule>]]></ac:plain-text-body></ac:structured-macro>')
    assert parsed.fragments[0].presentation == "A7 & B"
    assert "<rule>&amp;</rule>" in parsed.fragments[1].presentation


def test_offset_macro_instance_and_attribute_order_changes_preserve_material_match():
    native = '<p>Intro.</p><p><ac:structured-macro ac:name="status" ac:macro-id="old"><ac:parameter ac:name="title">Passed</ac:parameter></ac:structured-macro></p>'
    base = projection(native)
    old = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments if "Status: Passed" in f.presentation_text)
    target = projection('<p>Inserted elsewhere.</p>' + native.replace('ac:name="status" ac:macro-id="old"', 'ac:macro-id="new" ac:name="status"'), base)
    _, plan = planned(base, target, old)
    assert plan.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert plan.parts[0].current[0].anchor.range_start != old.anchor.range_start
    assert plan.parts[0].current[0].raw_content_sha256 != old.raw_content_sha256
    assert plan.route is SupportRoute.CHANGE_IMPACT


def test_header_changed_same_row_requires_whole_claim_impact_assessment():
    native = '<table><tr><th>Period</th><th>Allowed</th></tr><tr><td>Current</td><td>No</td></tr></table>'
    base = projection(native)
    old = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments if f.fragment_type.endswith("table-row"))
    target = projection(native.replace("Allowed", "Rejected"), base)
    context, plan = planned(base, target, old)
    assert plan.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert plan.route is SupportRoute.CHANGE_IMPACT
    assert "Rejected: No" in plan.parts[0].current[0].presentation_text
    assert context.delta_fragments()[0]


def coverage_target(base, coverage):
    if coverage == "complete_present":
        return base
    if coverage == "partial_carried":
        return replace(base, observations=(), coverage=ProjectionCoverage.PARTIAL_PROJECTION,
                       carried_observation_revision_ids=tuple(r.id for r in base.observation_revisions), deltas=())
    unit = replace(base.source_unit_revisions[0], id="removed-target", observation_revision_ids=())
    delta = replace(base.deltas[0], current_unit_revision_id=unit.id, added_observation_ids=(),
                    changed_anchors=(), removed_observation_ids=tuple(o.id for o in base.observations), fragment_mappings=())
    return replace(base, observations=(), observation_revisions=(), source_unit_revisions=(unit,), deltas=(delta,))


@pytest.mark.parametrize("coverage", ["complete_present", "complete_absent", "partial_carried"])
@pytest.mark.parametrize("corruption", ["origin", "kind", "presentation"])
def test_known_native_view_corruption_aborts_support_planning(corruption, coverage):
    base = projection('<table><tr><th>Rule</th></tr><tr><td>Approval required.</td></tr></table>')
    context = RevisionAssessmentContext(projection=coverage_target(base, coverage), base=base, access_context_hash="scope")
    row = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments if f.fragment_type.endswith("table-row"))
    prior = replace(support(base, row), text_view=row.text_view)
    assert prior.text_view.origins
    if corruption == "origin":
        origin, *rest = prior.text_view.origins
        prior = replace(prior, text_view=replace(prior.text_view,
            origins=(replace(origin, raw_content_sha256="0" * 64), *rest)))
    elif corruption == "kind":
        prior = replace(prior, text_view=replace(prior.text_view, kind="invented-kind"))
    else:
        prior = replace(prior, excerpt="Fabricated conclusion.")
    with pytest.raises(SupportRevalidationLimitation) as failure:
        plan_support_revision(context, [SupportWorkItem("w", memory(), (prior,), context)])
    assert failure.value.code.value == "evidence_integrity"


@pytest.mark.parametrize("coverage", ["complete_present", "complete_absent", "partial_carried"])
def test_unavailable_native_view_keeps_history_without_claiming_corruption(coverage):
    base = projection('<table><tr><th>Rule</th></tr><tr><td>Approval required.</td></tr></table>')
    context = RevisionAssessmentContext(projection=coverage_target(base, coverage), base=base, access_context_hash="scope")
    row = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments if f.fragment_type.endswith("table-row"))
    prior = replace(support(base, row), text_view=replace(row.text_view, version=99))
    result = plan_support_revision(context, [SupportWorkItem("w", memory(), (prior,), context)])
    assert result.supports[0].parts[0].status is EvidenceCorrespondence.UNKNOWN
    assert result.supports[0].route is SupportRoute.UNRESOLVED_PARTIAL_COVERAGE
    assert result.supports[0].parts[0].evidence is prior


def test_display_change_does_not_decide_cross_revision_correspondence():
    base = projection('<p>Stable rule.</p>')
    old = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments if f.presentation_text == "Stable rule.")
    target = projection('<p>Inserted.</p><p>Stable rule.</p>', base)
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash="scope")
    revision = target.observation_revisions[0]
    index = context.index(revision)
    changed = tuple(replace(f, presentation_text="Readable view: " + f.presentation_text,
                            presentation_sha256=hashlib.sha256(("Readable view: " + f.presentation_text).encode()).hexdigest()) for f in index.fragments)
    context.indexes[revision.id] = replace(index, fragments=changed)
    context.full_fragments = changed
    result = plan_support_revision(context, [SupportWorkItem("w", memory(), (support(base, old),), context)])
    assert result.supports[0].parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED


def test_entity_encoding_callback_boundaries_do_not_change_material():
    base = projection('<p>AB stays required.</p>')
    old = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments
               if f.presentation_text == "AB stays required.")
    target = projection('<p>Inserted.</p><p>A&#66; stays required.</p>', base)
    _, plan = planned(base, target, old)
    assert plan.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert plan.parts[0].current[0].presentation_text == old.presentation_text
    assert plan.parts[0].current[0].raw_content_sha256 != old.raw_content_sha256


def test_new_native_profile_does_not_rewrite_legacy_normalized_revision():
    native = '<p>Stable rule.</p>'
    current = projection(native)
    legacy = replace(current.observation_revisions[0], id="legacy", content="# Rules\n\nStable rule.",
                     evidence_profile=MARKDOWN_STRUCTURAL_PROFILE, semantic_hash="legacy-hash")
    old = replace(current, deltas=(), observation_revisions=(legacy,), source_unit_revisions=(replace(
        current.source_unit_revisions[0], id="legacy-unit", observation_revision_ids=(legacy.id,)),))
    target = projection(native, old)
    assert target.observation_revisions[0].id != legacy.id
    old_fragment = next(f for f in build_revision_fragment_index(legacy).fragments if f.presentation_text == "Stable rule.")
    _, plan = planned(old, target, old_fragment)
    assert plan.parts[0].status is EvidenceCorrespondence.MODIFIED
    assert plan.route is SupportRoute.SUPPORT_ASSESSMENT
    assert legacy.content == "# Rules\n\nStable rule."


@pytest.mark.parametrize("native", [
    '<p>unclosed',
    '<ac:structured-macro ac:name="unknown"><ac:parameter ac:name="x">value</ac:parameter></ac:structured-macro>',
    '<ac:structured-macro ac:name="status"><ac:parameter ac:name="title">Pass</ac:parameter><ac:parameter ac:name="title">Fail</ac:parameter></ac:structured-macro>',
])
def test_unsupported_or_ambiguous_native_structure_is_not_a_successful_partial_index(native):
    result = build_revision_fragment_index(projection('<p>Safe rule.</p>' + native).observation_revisions[0])
    assert any(error.fatal for error in result.errors)
    assert not result.fragments


def test_row_and_column_spans_preserve_stored_values_in_readable_rows():
    parsed = parse_storage('<table><tr><th>Case</th><th>Period</th><th>Result</th></tr><tr><td rowspan="2">14</td><td>Current</td><td>No</td></tr><tr><td colspan="2">Next period</td></tr></table>')
    assert parsed.fragments[-1].presentation == "Case: 14 [cell from row 2]\nPeriod / Result (columns 2–3): Next period"


def test_deleted_text_and_ordered_nested_lists_keep_authored_meaning():
    parsed = parse_storage('<p><del>Use A.</del> Use B.</p><ol start="3"><li>First<ul><li>Nested</li></ul></li><li value="7">Second</li></ol>')
    assert parsed.fragments[0].presentation == "[Deleted text: Use A.] Use B."
    assert parsed.fragments[1].presentation == "3. First\n  - Nested\n7. Second"


def test_table_content_outside_cells_is_rejected_instead_of_silently_lost():
    with pytest.raises(ValueError, match="unclassified content"):
        parse_storage('<table><tbody><p>Only for emergencies.</p><tr><td>Allowed</td></tr></tbody></table>')


def test_preformatted_whitespace_is_preserved():
    source = '<pre>    if x:\n        y()\n</pre>'
    parsed = parse_storage(source)
    assert parsed.fragments[0].presentation == "Preformatted text:\n    if x:\n        y()\n"


def test_container_macros_keep_table_structure_and_image_labels():
    parsed = parse_storage('<ac:structured-macro ac:name="info"><ac:rich-text-body><table><tr><th>Period</th><th>Allowed</th></tr><tr><td>Current</td><td>No</td></tr></table><ac:image ac:alt="Target design" ac:title="Proposed"><ri:attachment ri:filename="design.png"/></ac:image></ac:rich-text-body></ac:structured-macro>')
    text = parsed.fragments[0].presentation
    assert "Period: Current\nAllowed: No" in text
    assert "alt: Target design" in text and "title: Proposed" in text


def test_table_caption_qualifies_each_row_reading():
    projected = projection('<table><caption>Proposed cases only.</caption><tr><th>Period</th><th>Allowed</th></tr><tr><td>Current</td><td>No</td></tr></table>')
    context = RevisionAssessmentContext(projection=projected, base=None, access_context_hash="scope")
    row = next(f for f in context.full_fragments if f.fragment_type.endswith("table-row"))
    caption = next(f for f in context.full_fragments if f.fragment_type.endswith("caption"))
    assert caption.anchor in context.reading_context((row,))


def test_styled_wrapper_is_retained_in_its_complete_selection():
    parsed = parse_storage('<div style="color:red"><p>Use B.</p></div>')
    assert len(parsed.fragments) == 1
    assert "Use B." in parsed.fragments[0].presentation
    assert "style: color:red" in parsed.fragments[0].presentation


def test_native_directory_macro_omits_navigation_without_erasing_authored_siblings():
    native = ('<p>Generated by the deployment process.'
              '<ac:structured-macro ac:name="toc" ac:macro-id="directory">'
              '<ac:parameter ac:name="minLevel">2</ac:parameter>'
              '<ac:parameter ac:name="include">Eligibility.*</ac:parameter>'
              '</ac:structured-macro>Only for approved runs.</p>'
              '<h2>Page Properties</h2><p>Approval remains required.</p>')
    projected = projection(native)
    revision = projected.observation_revisions[0]
    index = build_revision_fragment_index(revision)
    assert not index.errors
    assert json.loads(revision.content)["body"] == native
    assert [f.presentation_text for f in index.fragments[:-1]] == [
        "Generated by the deployment process.\nOnly for approved runs.",
        "Page Properties", "Approval remains required.",
    ]
    assert index.fragments[0].anchor.range_start is not None
    assert index.fragments[0].raw_content_sha256 == hashlib.sha256(
        revision.content[index.fragments[0].anchor.range_start:index.fragments[0].anchor.range_end].encode()
    ).hexdigest()


def test_native_directory_only_container_is_not_authored_evidence():
    parsed = parse_storage('<p><ac:structured-macro ac:name="toc"/></p><h1>Scope</h1><p>Only after approval.</p>')
    assert [f.presentation for f in parsed.fragments] == ["Scope", "Only after approval."]


def test_hidden_issue_origin_change_is_not_same_material_even_when_display_is_unchanged():
    native = ('<p><ac:structured-macro ac:name="jira">'
              '<ac:parameter ac:name="key">PAY-42</ac:parameter>'
              '<ac:parameter ac:name="serverId">first-origin</ac:parameter>'
              '<ac:parameter ac:name="columnIds">issuekey,summary</ac:parameter>'
              '</ac:structured-macro></p>')
    base = projection(native)
    old = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments
               if f.presentation_text == "Issue: PAY-42")
    assert "first-origin" not in old.presentation_text
    target = projection(native.replace("first-origin", "second-origin"), base)
    _, plan = planned(base, target, old)
    assert plan.parts[0].status is EvidenceCorrespondence.MODIFIED
    current = next(f for f in build_revision_fragment_index(target.observation_revisions[0]).fragments
                   if f.presentation_text == old.presentation_text)
    assert current.content_value != old.content_value


@pytest.mark.parametrize("body", [
    '<ac:rich-text-body><p>Hidden rule.</p></ac:rich-text-body>',
    '<ac:parameter ac:name="unsupported">Only after approval.</ac:parameter>',
])
def test_undeclared_directory_content_does_not_silently_disappear(body):
    index = build_revision_fragment_index(projection(
        '<p>Safe rule.</p><ac:structured-macro ac:name="toc">' + body + '</ac:structured-macro>'
    ).observation_revisions[0])
    assert not index.fragments and any(error.fatal for error in index.errors)
