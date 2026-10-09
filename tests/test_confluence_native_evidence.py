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
    '<p><em>crossed</p></em>',
    '<p>&undefined;</p>',
    '<p a="1" a="2">Twice</p>',
    '<ac:structured-macro ac:name="code"><ac:plain-text-body><![CDATA[cut off</ac:plain-text-body></ac:structured-macro>',
    '<ac:layout><p>Outside a section.</p></ac:layout>',
])
def test_malformed_native_structure_is_not_a_successful_partial_index(native):
    result = build_revision_fragment_index(projection('<p>Safe rule.</p>' + native).observation_revisions[0])
    assert any(error.fatal for error in result.errors)
    assert not result.fragments


@pytest.mark.parametrize("macro, expected", [
    ('<ac:structured-macro ac:name="children"/>', "[Macro children]"),
    ('<ac:structured-macro ac:name="diagram" ac:macro-id="m"><ac:parameter ac:name="diagramName">Payment flow</ac:parameter>'
     '<ac:parameter ac:name="revision">7</ac:parameter><ac:parameter ac:name="">default</ac:parameter>'
     '<ac:parameter ac:name="empty"></ac:parameter></ac:structured-macro>',
     "[Macro diagram: diagramName=Payment flow; revision=7; default]"),
    ('<ac:structured-macro ac:name="tree"><ac:parameter ac:name="root"><ac:link><ri:page ri:content-title="Home"/></ac:link>'
     '</ac:parameter></ac:structured-macro>', "[Macro tree: root=[page: content-title=Home]]"),
    ('<ac:structured-macro ac:name="status"><ac:parameter ac:name="title">Pass</ac:parameter>'
     '<ac:parameter ac:name="subtle">true</ac:parameter></ac:structured-macro>', "[Macro status: title=Pass; subtle=true]"),
    ('<ac:structured-macro ac:name="status"><ac:parameter ac:name="title">Pass</ac:parameter>'
     '<ac:parameter ac:name="title">Fail</ac:parameter></ac:structured-macro>', "[Macro status: title=Pass; title=Fail]"),
    ('<ac:structured-macro ac:name="jira"><ac:parameter ac:name="jqlQuery">project = PAY</ac:parameter></ac:structured-macro>',
     "[Macro jira: jqlQuery=project = PAY]"),
])
def test_macro_outside_a_declared_rule_states_its_own_envelope_and_nothing_else(macro, expected):
    parsed = parse_storage('<p>Safe rule.</p>' + macro)
    assert [fragment.presentation for fragment in parsed.fragments] == ["Safe rule.", expected]
    changed = parse_storage('<p>Safe rule.</p>' + macro.replace('ac:name="', 'ac:name="x', 1))
    assert changed.fragments[1].content_value != parsed.fragments[1].content_value


def test_undeclared_plain_text_macro_body_is_literal():
    parsed = parse_storage('<ac:structured-macro ac:name="uml"><ac:parameter ac:name="format">svg</ac:parameter>'
                           '<ac:plain-text-body><![CDATA[A -> B : <pay>\n  note right]]></ac:plain-text-body></ac:structured-macro>')
    assert [fragment.presentation for fragment in parsed.fragments] == ["Macro uml\nformat: svg\nA -> B : <pay>\n  note right\n"]


def test_undeclared_container_macro_keeps_blocks_selectable_under_its_parameters():
    native = ('<h1>Overview</h1><ac:structured-macro ac:name="properties" ac:macro-id="m">'
              '<ac:parameter ac:name="id">release</ac:parameter><ac:rich-text-body>'
              '<h2>Inside</h2><table><tr><th>Key</th><th>Value</th></tr><tr><td>Owner</td><td>Payroll</td></tr></table>'
              '<p>Only after approval.</p></ac:rich-text-body></ac:structured-macro><p>After the container.</p>')
    projected = projection(native)
    context = RevisionAssessmentContext(projection=projected, base=None, access_context_hash="scope")
    by_text = {fragment.presentation_text: fragment for fragment in context.full_fragments}
    assert {"Macro properties id: release", "Inside", "Key: Owner\nValue: Payroll", "Only after approval.",
            "After the container."} <= by_text.keys()
    setting = by_text["Macro properties id: release"].anchor
    assert setting in context.reading_context((by_text["Only after approval."],))
    assert setting not in context.reading_context((by_text["After the container."],))
    inside = by_text["Inside"].anchor
    assert inside in context.reading_context((by_text["Only after approval."],))
    assert inside not in context.reading_context((by_text["After the container."],))
    assert by_text["Overview"].anchor in context.reading_context((by_text["After the container."],))


def test_inline_only_container_body_stays_one_selection():
    parsed = parse_storage('<ac:structured-macro ac:name="tooltip"><ac:parameter ac:name="tip">Net of tax</ac:parameter>'
                           '<ac:rich-text-body>Gross <strong>pay</strong></ac:rich-text-body></ac:structured-macro>')
    assert [fragment.presentation for fragment in parsed.fragments] == ["Macro tooltip\ntip: Net of tax\nGross pay"]


def test_editor_instructions_and_opaque_identities_are_not_page_content():
    parsed = parse_storage(
        '<p><ac:placeholder>Type the decision here</ac:placeholder></p>'
        '<p>Decided: <ac:placeholder ac:type="mention">@owner</ac:placeholder>ship it '
        '<ac:emoticon ac:name="tick" ac:emoji-shortname=":check_mark:" ac:emoji-id="atlassian-check_mark"/></p>'
        '<ac:task-list><ac:task><ac:task-id>4</ac:task-id><ac:task-uuid>0e1f</ac:task-uuid>'
        '<ac:task-status>incomplete</ac:task-status><ac:task-body>Approve rollout.</ac:task-body></ac:task></ac:task-list>')
    assert [fragment.presentation for fragment in parsed.fragments] == [
        "Decided: ship it [Emoticon: tick]", "Task (incomplete): Approve rollout."]
    renamed = parse_storage('<p>Decided: ship it <ac:emoticon ac:name="tick" ac:emoji-id="other"/></p>')
    assert renamed.fragments[0].content_value != parsed.fragments[0].content_value


def test_elements_and_placements_outside_the_documented_grammar_keep_their_text():
    parsed = parse_storage(
        '<ac:layout><ac:layout-section ac:type="four_equal"><ac:layout-cell>Loose &amp; unwrapped<p>Wrapped.</p></ac:layout-cell>'
        '</ac:layout-section></ac:layout><small>Fine print.</small>'
        '<ul><li>First</li><ul><li>Nested directly</li></ul><li>Second</li></ul>')
    assert [(fragment.kind, fragment.presentation) for fragment in parsed.fragments] == [
        ("text", "Loose & unwrapped"), ("p", "Wrapped."), ("small", "Fine print."),
        ("ul", "- First\n  - Nested directly\n- Second"),
    ]
    source = '<ac:layout><ac:layout-section ac:type="four_equal"><ac:layout-cell>Loose &amp; unwrapped'
    assert source[parsed.fragments[0].start:parsed.fragments[0].end] == "Loose &amp; unwrapped"


def test_table_inside_a_cell_is_that_cells_content():
    parsed = parse_storage('<table><tr><th>Case</th><th>Detail</th></tr><tr><td>14</td><td>'
                           '<table><tr><th>Period</th><th>Allowed</th></tr><tr><td>Current</td><td>No</td></tr></table>'
                           '</td></tr></table>')
    assert [fragment.kind for fragment in parsed.fragments] == ["table-header", "table-row"]
    assert parsed.fragments[1].presentation == "Case: 14\nDetail: Row 1:\nPeriod: Period\nAllowed: Allowed\nRow 2:\nPeriod: Current\nAllowed: No"


def test_table_without_a_rectangular_grid_is_one_whole_selection():
    parsed = parse_storage('<p>Rule.</p><table><tr><th>Case</th><th>Period</th></tr><tr><td>14</td></tr>'
                           '<tr><td rowspan="9">15</td><td>Next</td></tr></table>')
    assert [(fragment.kind, fragment.presentation) for fragment in parsed.fragments] == [
        ("p", "Rule."), ("table", "Case | Period\n14\n15 | Next")]


def test_row_and_column_spans_preserve_stored_values_in_readable_rows():
    parsed = parse_storage('<table><tr><th>Case</th><th>Period</th><th>Result</th></tr><tr><td rowspan="2">14</td><td>Current</td><td>No</td></tr><tr><td colspan="2">Next period</td></tr></table>')
    assert parsed.fragments[-1].presentation == "Case: 14 [cell from row 2]\nPeriod / Result (columns 2–3): Next period"


def test_deleted_text_and_ordered_nested_lists_keep_authored_meaning():
    parsed = parse_storage('<p><del>Use A.</del> Use B.</p><ol start="3"><li>First<ul><li>Nested</li></ul></li><li value="7">Second</li></ol>')
    assert parsed.fragments[0].presentation == "[Deleted text: Use A.] Use B."
    assert parsed.fragments[1].presentation == "3. First\n  - Nested\n7. Second"


def test_table_content_outside_cells_stays_in_one_whole_table_selection():
    parsed = parse_storage('<table><tbody><p>Only for emergencies.</p><tr><td>Allowed</td></tr></tbody></table>')
    assert [(fragment.kind, fragment.presentation) for fragment in parsed.fragments] == [
        ("table", "Only for emergencies.\nAllowed")]


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


@pytest.mark.parametrize("body, expected", [
    ('<ac:rich-text-body><p>Hidden rule.</p></ac:rich-text-body>', "Hidden rule."),
    ('<ac:parameter ac:name="unsupported">Only after approval.</ac:parameter>', "[Macro toc: unsupported=Only after approval.]"),
])
def test_undeclared_directory_content_does_not_silently_disappear(body, expected):
    index = build_revision_fragment_index(projection(
        '<p>Safe rule.</p><ac:structured-macro ac:name="toc">' + body + '</ac:structured-macro>'
    ).observation_revisions[0])
    assert not index.errors
    assert expected in [fragment.presentation_text for fragment in index.fragments]
