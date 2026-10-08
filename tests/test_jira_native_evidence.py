"""Jira's attested current rendering, literal history and revision correspondence."""

import hashlib
import json
from dataclasses import replace

import pytest

from memforge.pipeline.evidence_fragments import build_revision_fragment_index
from memforge.pipeline.evidence_fragments import verify_text_evidence_view
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.pipeline.support_reading import EvidenceCorrespondence, SupportRoute
from memforge.source_adapters.jira import (
    CANONICAL_RECORD_SCHEMAS,
    CURRENT_OBSERVATION_PROFILES,
    LEGACY_OBSERVATION_PROFILES,
    changelog_record,
    changelog_semantic_class,
    comment_record,
    core_revised_at,
    issue_record,
)
from memforge.source_adapters.jira_html import parse_rendered_html
from tests.test_confluence_native_evidence import planned
from tests.test_source_projection_adapters import _inputs, _item, _jira_payload


def projection(native, html, *, prior=None, comments=None, histories=None, fields=None):
    item = _item(item_id="jira-PAY-12", title="Payroll", extra={"issue_id": "10012"})
    payload = _jira_payload(field_overrides={"description": native, **(fields or {})}, comments=comments, histories=histories)
    payload["renderedFields"] = {"description": html}
    raw, body = _inputs(item, payload, markdown="INTENTIONALLY WRONG NORMALIZATION")
    return project_source_item(
        source_id="src", source_type="jira", run_id="r2" if prior else "r1",
        item=item, raw=raw, normalized=body,
        prior_unit_revision=prior.source_unit_revisions[0] if prior else None,
        prior_observation_revisions={r.observation_id: r for r in prior.observation_revisions} if prior else None,
    )


@pytest.mark.parametrize("html, expected", [
    ('<p><span class="nobr"><a href="https://jira.example.test/browse/PAY-1">PAY-1</a><img class="rendericon" src="/images/link.gif" height="7" width="7" align="absmiddle" alt="" border="0"></span></p>', 'PAY-1'),
    ('<p><a class="issue-link" data-issue-key="PAY-1" href="https://jira.example.test/browse/PAY-1">PAY-1</a></p>', 'PAY-1'),
    ('<a name="code-start"></a><pre>Keep the literal example.</pre>', 'Keep the literal example.'),
    ('<p><img src="/secure/attachment/1/image.png" style="border: 0px solid black" width="200" height="100" alt="Example"></p>', 'Example'),
    ('<ul class="alternate" type="square"><li>Keep the condition.</li></ul>', 'Keep the condition.'),
    ('<p><ins>Added condition.</ins> <cite>Source title.</cite></p>', 'Source title.'),
    ('<p><span class="error">The supplied reference cannot be resolved.</span></p>', 'cannot be resolved'),
    ('<pre class="code-java"><span class="code-object">Object</span><span class="code-quote-red">"value"</span></pre>', 'Object"value"'),
    ('<p>First<br class="atl-forced-newline">Second</p>', 'First\nSecond'),
])
def test_common_jira_rendering_retains_authored_text(html, expected):
    parsed = parse_rendered_html(html)
    assert any(expected in fragment.presentation for fragment in parsed.fragments)
    assert all(fragment.presentation != "()" for fragment in parsed.fragments)
    for fragment in parsed.fragments:
        assert html[fragment.start:fragment.end]


def test_renderer_link_icons_are_decoration_but_image_and_link_identities_are_material():
    simple = '<p><a href="https://example.test/doc">Reference</a></p>'
    decorated = '<p><span class="nobr"><a href="https://example.test/doc">Reference</a><img class="rendericon" src="/link.gif" height="7" width="7" align="absmiddle" alt="" border="0"></span></p>'
    assert parse_rendered_html(simple).fragments[0].content_value == parse_rendered_html(decorated).fragments[0].content_value
    plain = '<p>Keep A.</p>'
    bookmarked = '<p><a name="generated-1"></a>Keep A.</p>'
    assert parse_rendered_html(plain).fragments[0].content_value == parse_rendered_html(bookmarked).fragments[0].content_value
    assert parse_rendered_html(bookmarked).fragments[0].content_value == parse_rendered_html(bookmarked.replace('generated-1', 'generated-2')).fragments[0].content_value
    image = '<p><img src="/attachment/1" alt="Example" width="200" height="100" style="border: 0px solid black"></p>'
    resized = image.replace('width="200"', 'width="400"')
    assert parse_rendered_html(image).fragments[0].content_value == parse_rendered_html(resized).fragments[0].content_value
    assert parse_rendered_html(image).fragments[0].content_value != parse_rendered_html(image.replace('/attachment/1', '/attachment/2')).fragments[0].content_value
    inserted = '<p><ins>Changed</ins></p>'
    assert parse_rendered_html(inserted).fragments[0].content_value != parse_rendered_html('<p>Changed</p>').fragments[0].content_value
    assert parse_rendered_html(inserted).fragments[0].presentation == 'Changed'
    assert parse_rendered_html('<p><cite>Title</cite></p>').fragments[0].content_value != parse_rendered_html('<p>Title</p>').fragments[0].content_value
    labelled = parse_rendered_html('<img class="rendericon" src="/attachment/1" alt="Material condition">').fragments[0]
    assert labelled.presentation == 'Material condition (image: /attachment/1)'
    assert 'rendericon' in labelled.content_value


def test_legacy_jira_schema_upgrade_plans_current_reading_and_next_revision_stays_incremental():
    from memforge.pipeline.projection_context import CommittedSourceUnitSnapshot, ExtractionAuthority, plan_projection_evidence_work

    base = projection("Keep A.", "<p>Keep A.</p>")
    core = base.observation_revisions[0]
    old_record = json.loads(core.content)
    for key in ("native_description", "representation", "issue_key", "issue_id"):
        old_record.pop(key, None)
    old_record["description"] = "Keep A."
    old_content = json.dumps(old_record, sort_keys=True)
    legacy = replace(core, id="legacy-core", content=old_content,
        semantic_hash=hashlib.sha256(old_content.encode()).hexdigest(),
        evidence_profile=LEGACY_OBSERVATION_PROFILES[("jira", "issue_core")])
    legacy_unit = replace(base.source_unit_revisions[0], id="legacy-unit", observation_revision_ids=(legacy.id,))
    old = replace(base, observation_revisions=(legacy,), source_unit_revisions=(legacy_unit,), deltas=())
    target = projection("Keep A.", "<p>Keep A.</p>", prior=old)
    authority = plan_projection_evidence_work(target,
        committed_base_snapshot=CommittedSourceUnitSnapshot(legacy_unit, (legacy,)),
        reprocess_all_current_observations=False)
    assert isinstance(authority, ExtractionAuthority)
    fragment = next(f for f in build_revision_fragment_index(target.observation_revisions[0]).fragments if f.presentation_text.endswith("Keep A."))
    assert authority.authorizes(fragment)
    historical = next(f for f in build_revision_fragment_index(legacy).fragments if f.presentation_text.endswith("Keep A."))
    _, support = planned(old, target, historical)
    assert support.parts[0].status is EvidenceCorrespondence.MODIFIED
    assert support.route is not SupportRoute.REBIND_SUPPORT

    subsequent = projection("Keep A. Add B.", "<p>Keep A.</p><p>Add B.</p>", prior=target)
    next_plan = plan_projection_evidence_work(subsequent,
        committed_base_snapshot=CommittedSourceUnitSnapshot(target.source_unit_revisions[0], target.observation_revisions),
        reprocess_all_current_observations=False)
    assert isinstance(next_plan, ExtractionAuthority)
    next_fragments = build_revision_fragment_index(subsequent.observation_revisions[0]).fragments
    kept = next(f for f in next_fragments if f.presentation_text.endswith("Keep A."))
    added = next(f for f in next_fragments if f.presentation_text.endswith("Add B."))
    assert not next_plan.authorizes(kept)
    assert next_plan.authorizes(added)


def test_current_html_is_attested_and_native_value_is_retained_without_duplicate_extraction():
    native = '{code:html}<form role="region"><layout sap{code}'
    html = '<p>Only after approval.</p><pre class="code-html">&lt;form role="region"&gt;&lt;layout sap</pre>'
    projected = projection(native, html)
    revision = projected.observation_revisions[0]
    record = json.loads(revision.content)
    assert record["native_description"] == native
    assert record["description"] == html
    assert record["representation"] == "jira-issue-core:5"
    index = build_revision_fragment_index(revision)
    assert not index.errors
    views = [f.presentation_text for f in index.fragments]
    assert any(view.endswith('Code (language: html):\n<form role="region"><layout sap') for view in views)
    assert all("{code}" not in view and "WRONG NORMALIZATION" not in view for view in views)
    assert sum(view.endswith("Only after approval.") for view in views) == 1
    paragraph = next(f for f in index.fragments if f.presentation_text.endswith("Only after approval."))
    assert paragraph.text_view is not None and paragraph.text_view.origins
    for fragment in index.fragments:
        anchor = fragment.anchor
        assert hashlib.sha256(revision.content[anchor.range_start:anchor.range_end].encode()).hexdigest() == fragment.raw_content_sha256


def test_syntax_highlighting_offsets_and_unrelated_native_edits_keep_unique_code_correspondence():
    native = '{code:html}<form role="region"><layout sap{code}'
    old_html = '<p>Scope.</p><pre class="code-html"><span class="code-tag">&lt;form role=<span class="code-quote">"region"</span>&gt;</span>&lt;layout sap</pre>'
    base = projection(native, old_html)
    old = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments if f.fragment_type.endswith("pre"))
    new_html = '<p>New unrelated paragraph.</p><p>Scope.</p><pre class="code-html">&lt;form role="region"&gt;&lt;layout sap</pre>'
    target = projection("New paragraph. " + native, new_html, prior=base)
    _, plan = planned(base, target, old)
    assert plan.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    rebound = plan.parts[0].current[0]
    assert rebound.anchor.range_start != old.anchor.range_start
    assert rebound.raw_content_sha256 != old.raw_content_sha256
    assert rebound.presentation_text == old.presentation_text


def test_literal_history_is_not_rendered_as_current_html_or_markdown():
    old = '{code:html}<form role="region"><layout old{code}'
    new = '{code:html}<form role="region"><layout new{code}'
    history = {"id": "9", "created": "2026-01-01T12:00:00Z", "items": [
        {"field": "description", "fromString": old, "toString": new},
    ]}
    projected = projection(new, '<pre>&lt;form role="region"&gt;&lt;layout new</pre>', histories=[history])
    revision = next(r for r in projected.observation_revisions if r.evidence_profile.schema_name == "jira-changelog")
    index = build_revision_fragment_index(revision)
    assert not index.errors
    assert any(f.presentation_text.endswith(old) for f in index.fragments)
    assert any(f.presentation_text.endswith(new) for f in index.fragments)
    assert "previous literal native" in CANONICAL_RECORD_SCHEMAS[("jira-changelog", 2)].model_interpretation
    assert json.loads(revision.content)["items"] == history["items"]


def test_history_views_bind_readable_event_roles_and_preserve_old_schema_views():
    author = {"displayName": "Example Author", "key": "USER-9", "avatarUrls": {"large": "https://example.test/avatar"}}
    history = {"id": "9", "created": "2026-01-01T12:00:00Z", "author": author,
               "items": [{"field": "description", "fromString": "Use A.", "toString": "Use B."}]}
    base = projection("Use B.", "<p>Use B.</p>", histories=[history])
    revision = next(r for r in base.observation_revisions if r.evidence_profile.schema_name == "jira-changelog")
    selected = next(f for f in build_revision_fragment_index(revision).fragments
                    if f.presentation_text.endswith("New value:\nUse B."))
    assert "Changed field: description" in selected.presentation_text
    assert "Change recorded: 2026-01-01T12:00:00Z" in selected.presentation_text
    assert "Change author: Example Author" in selected.presentation_text
    assert "avatarUrls" not in selected.presentation_text and "https://example.test/avatar" not in selected.presentation_text
    assert selected.text_view.origins
    verify_text_evidence_view(revision, anchor=selected.anchor, text_view=selected.text_view,
                             raw_content_sha256=selected.raw_content_sha256,
                             presentation_sha256=selected.presentation_sha256, excerpt=selected.presentation_text)
    legacy = replace(revision, evidence_profile=replace(revision.evidence_profile, schema_version=3))
    historical = next(f for f in build_revision_fragment_index(legacy).fragments if f.presentation_text.endswith("Use B."))
    assert "avatarUrls" in historical.presentation_text and "New value:" not in historical.presentation_text
    verify_text_evidence_view(legacy, anchor=historical.anchor, text_view=historical.text_view,
                             raw_content_sha256=historical.raw_content_sha256,
                             presentation_sha256=historical.presentation_sha256, excerpt=historical.presentation_text)


def test_history_item_reorder_corresponds_but_changed_interpretation_is_assessed():
    history = {"id": "9", "created": "2026-01-01T12:00:00Z", "items": [
        {"field": "description", "fromString": "Use A.", "toString": "Use B."},
        {"field": "summary", "fromString": "Earlier heading", "toString": "New heading"},
    ]}
    base = projection("Use B.", "<p>Use B.</p>", histories=[history])
    revision = next(r for r in base.observation_revisions if r.evidence_profile.schema_name == "jira-changelog")
    old = next(f for f in build_revision_fragment_index(revision).fragments if f.presentation_text.endswith("New value:\nUse B."))
    reordered = {**history, "items": list(reversed(history["items"]))}
    target = projection("Use B.", "<p>Use B.</p>", prior=base, histories=[reordered])
    context, plan = planned(base, target, old)
    assert plan.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert plan.parts[0].current[0].anchor.range_start != old.anchor.range_start
    assert plan.parts[0].current[0].presentation_text == old.presentation_text
    from tests.test_text_view_contract import authority as planned_authority
    authority = planned_authority(base, target)
    history_id = old.anchor.observation_id
    assert not any(authority.authorizes(f) for f in context.full_fragments
                   if f.anchor.observation_id == history_id)
    current, removed = context.delta_fragments()
    assert not any(f.anchor.observation_id == history_id for f in current)
    assert not any(f.anchor.observation_id == history_id for f in removed)
    changed = {**history, "items": [{**history["items"][0], "field": "policy"}, history["items"][1]]}
    target = projection("Use B.", "<p>Use B.</p>", prior=base, histories=[changed])
    context, plan = planned(base, target, old)
    assert plan.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert plan.route is SupportRoute.CHANGE_IMPACT
    assert "Changed field: policy" in plan.parts[0].current[0].presentation_text
    assert context.delta_fragments()[0]


def test_scalar_native_transport_change_uses_declared_comparison_for_match_and_work():
    from tests.test_text_view_contract import authority
    base = projection("Rule", "<p>Rule</p>", fields={"status": {"id": "1", "name": "Open", "iconUrl": "old"}})
    old = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments
               if f.presentation_text.splitlines()[-1].startswith("Status: "))
    target = projection("Rule", "<p>Rule</p>", prior=base,
                        fields={"status": {"id": "1", "name": "Open", "iconUrl": "new"}})
    context, plan = planned(base, target, old)
    assert plan.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert old.raw_content_sha256 != plan.parts[0].current[0].raw_content_sha256
    assert "iconUrl" not in old.presentation_text
    assert not authority(base, target).authorizes(plan.parts[0].current[0])
    assert context.delta_fragments() == ((), ())


def test_comment_rendered_body_and_attachment_identity_survive_projection():
    comment = {"id": "20", "body": "*Not resolved*", "renderedBody": '<p><del>Resolved</del> Still pending.</p>', "attachments": [{"id": "33"}]}
    projected = projection(None, None, comments=[comment])
    revision = next(r for r in projected.observation_revisions if r.evidence_profile.schema_name == "jira-comment")
    record = json.loads(revision.content)
    assert record["native_body"] == comment["body"]
    assert record["attachments"] == [{"id": "33"}]
    index = build_revision_fragment_index(revision)
    assert not index.errors
    assert any(f.presentation_text.endswith("[struck through: Resolved] Still pending.") for f in index.fragments)


@pytest.mark.parametrize("rendered", [None, 42, "", "   "])
def test_nonempty_current_native_without_explicit_html_is_rejected(rendered):
    with pytest.raises(ValueError, match="explicit provider-rendered HTML"):
        issue_record({"fields": {"description": "Native body"}, "renderedFields": {"description": rendered}})
    with pytest.raises(ValueError, match="explicit provider-rendered HTML"):
        comment_record({"body": "Native body", "renderedBody": rendered})


@pytest.mark.parametrize("native", [None, ""])
def test_truly_absent_native_fields_need_no_rendering(native):
    assert issue_record({"fields": {"description": native}})["description"] == native
    assert comment_record({"body": native})["body"] == native


def test_unmatched_rendering_or_nonstring_native_is_not_a_fallback():
    with pytest.raises(ValueError, match="no matching native"):
        issue_record({"fields": {}, "renderedFields": {"description": "<p>Stale</p>"}})
    with pytest.raises(ValueError, match="string native"):
        comment_record({"body": {"type": "doc"}, "renderedBody": "<p>Text</p>"})
    with pytest.raises(ValueError, match="literal native strings"):
        changelog_record({"items": [{"fromString": {"doc": "value"}}]})
    with pytest.raises(ValueError, match="no selectable view"):
        issue_record({"fields": {"description": "Missing meaningful value"}, "renderedFields": {"description": "<p> </p>"}})


def test_inline_literal_code_whitespace_and_link_label_are_preserved():
    fragment = parse_rendered_html('<p>Use <tt>a  b &lt;x&gt;</tt> at <a title="Other title" href="/guide">the guide</a>.</p>').fragments[0]
    assert fragment.presentation == "Use a  b <x> at the guide (/guide; title: Other title)."
    changed = parse_rendered_html('<p>Use <tt>a b &lt;x&gt;</tt> at <a title="Other title" href="/guide">the guide</a>.</p>').fragments[0]
    assert changed.content_value != fragment.content_value


def test_link_destination_and_strikethrough_remain_meaningful_comparison_material():
    source = '<p><del>Use A.</del> Use <a href="https://example.test/B">B</a>.</p>'
    original = parse_rendered_html(source).fragments[0]
    assert "[struck through: Use A.]" in original.presentation
    assert "https://example.test/B" in original.presentation
    assert parse_rendered_html(source.replace("example.test/B", "example.test/C")).fragments[0].content_value != original.content_value
    assert parse_rendered_html(source.replace("<del>", "<span>").replace("</del>", "</span>")).fragments[0].content_value != original.content_value


def test_inline_word_boundary_change_requires_changed_evidence_correspondence():
    base = projection("A B C", '<p>A<b> B </b>C</p>')
    old = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments if f.presentation_text.endswith("A B C"))
    target = projection("ABC", '<p>A<b>B</b>C</p>', prior=base)
    _, plan = planned(base, target, old)
    assert plan.parts[0].status is not EvidenceCorrespondence.EXACT_UNCHANGED
    assert old.content_value != next(f.content_value for f in build_revision_fragment_index(target.observation_revisions[0]).fragments if f.presentation_text.endswith("ABC"))


def test_attachment_preview_keeps_identity_and_does_not_invent_image_content():
    source = '<p><span class="image-wrap" style=""><a id="33_thumb" href="/attachment/33/design.png" file-preview-id="33" file-preview-title="design.png"><img src="/thumbnail/33.png" role="presentation" style="border: 0px solid black"/></a></span></p>'
    fragment = parse_rendered_html(source).fragments[0]
    assert fragment.presentation == "design.png (/attachment/33/design.png; attachment: 33)"
    assert parse_rendered_html(source.replace('id="33_thumb" ', "")).fragments[0].content_value == fragment.content_value
    changed = source.replace('file-preview-id="33"', 'file-preview-id="34"').replace('id="33_thumb"', 'id="34_thumb"')
    assert parse_rendered_html(changed).fragments[0].content_value != fragment.content_value
    with_caption = source.replace('<img ', 'Before correction: <img ')
    assert "Before correction:" in parse_rendered_html(with_caption).fragments[0].presentation
    with_alt = source.replace('<img ', '<img alt="Draft arrangement" ')
    assert "Draft arrangement" in parse_rendered_html(with_alt).fragments[0].presentation


def test_entity_spelling_and_attribute_order_do_not_change_values():
    a = parse_rendered_html('<p>A&#66; <a href="/rule" title="Rule">Rule</a></p>').fragments[0]
    b = parse_rendered_html('<p>AB <a title="Rule" href="/rule">Rule</a></p>').fragments[0]
    assert a.content_value == b.content_value


def test_table_headers_and_ordered_list_numbering_remain_readable_context():
    source = '<table><tr><th>Period</th><th>Allowed</th></tr><tr><td>Current</td><td>No</td></tr></table><ol start="3"><li>Approval</li><li value="7">Execute</li></ol>'
    parsed = parse_rendered_html(source)
    assert parsed.fragments[1].presentation == "Period: Current\nAllowed: No"
    assert parsed.groups[0].context_ranges == ((parsed.fragments[0].start, parsed.fragments[0].end),)
    assert parsed.fragments[2].presentation == "3. Approval\n7. Execute"


@pytest.mark.parametrize("source", [
    '<p>Open', '<p><em>Wrong nesting</p></em>', '<p a="x" a="y">Duplicated attribute</p>',
    '<p>&undefined;</p>', '<?php echo 1 ?><p>Visible</p>',
])
def test_malformed_html_fails_as_a_whole(source):
    with pytest.raises(ValueError):
        parse_rendered_html(source)
    with pytest.raises(ValueError):
        issue_record({"fields": {"description": "Native"}, "renderedFields": {"description": source}})


@pytest.mark.parametrize("source, plain, expected", [
    ('<p><span style="color:red">Red means rejected.</span></p>', '<p><span>Red means rejected.</span></p>',
     'Red means rejected. [style: color:red]'),
    ('<p><span style="display:none">Hidden exception</span>Visible rule</p>', '<p><span>Hidden exception</span>Visible rule</p>',
     'Hidden exception [style: display:none]Visible rule'),
    ('<p hidden="hidden">Hidden rule</p>', '<p>Hidden rule</p>', 'Hidden rule [hidden: hidden]'),
    ('<p hidden>Hidden rule</p>', '<p>Hidden rule</p>', 'Hidden rule [hidden]'),
    ('<p><span class="unknown-widget">Widget interpretation</span></p>', '<p><span>Widget interpretation</span></p>',
     'Widget interpretation [class: unknown-widget]'),
    ('<p><kbd>Ctrl</kbd> then confirm</p>', '<p>Ctrl then confirm</p>', 'Ctrl then confirm'),
    ('<a href="/rule" rel="alternate" target="rules">Other rule</a>', '<a href="/rule">Other rule</a>',
     'Other rule (/rule) [rel: alternate; target: rules]'),
    ('<div class="panel" style="background-color: #ffc;border-width: 1px;"><div class="panelContent"><p>Only after approval.</p></div></div>',
     '<div class="panel" style="border-width: 1px;"><div class="panelContent"><p>Only after approval.</p></div></div>',
     'Only after approval. [style: background-color: #ffc;border-width: 1px;]'),
    ('<font color="red" data-origin="paste">Rejected</font>', '<font color="red">Rejected</font>',
     'Rejected [color: red] [data-origin: paste]'),
])
def test_undeclared_rendering_vocabulary_stays_visible_and_compared(source, plain, expected):
    parsed = parse_rendered_html(source)
    assert [fragment.presentation for fragment in parsed.fragments] == [expected]
    assert parsed.fragments[0].content_value != parse_rendered_html(plain).fragments[0].content_value
    assert issue_record({"fields": {"description": "Native"}, "renderedFields": {"description": source}})["description"] == source


def test_content_html_never_displays_is_not_authored_text():
    parsed = parse_rendered_html('<!--hidden--><p>Visible</p><script>ignore()</script><style>p { color: red }</style>')
    assert [fragment.presentation for fragment in parsed.fragments] == ['Visible']
    with pytest.raises(ValueError, match="no selectable view"):
        issue_record({"fields": {"description": "Native"}, "renderedFields": {"description": '<script>ignore()</script>'}})


@pytest.mark.parametrize("table, expected", [
    ('<table><tr><th>Case</th><th>Period</th></tr><tr><td colspan="2">Applies to both.</td></tr></table>',
     'Case | Period\nApplies to both.'),
    ('<table><tbody><tr><td>14</td><td><div class="table-wrap"><table><tr><td>Current</td><td>No</td></tr></table></div></td></tr></tbody></table>',
     '14 | Current | No'),
    ('<table><tr><th>Case</th><th>Period</th></tr><tr><td>14</td></tr></table>', 'Case | Period\n14'),
])
def test_table_without_a_simple_grid_is_one_whole_selection(table, expected):
    parsed = parse_rendered_html('<p>Rule.</p><div class="table-wrap">' + table + '</div>')
    assert [(fragment.kind, fragment.presentation) for fragment in parsed.fragments] == [('p', 'Rule.'), ('table', expected)]


def test_list_content_between_items_stays_in_reading_order():
    parsed = parse_rendered_html('<ul><li>First</li><ul><li>Nested directly</li></ul><li>Second</li></ul>')
    assert parsed.fragments[0].presentation == '• First\n• Nested directly\n• Second'


def test_schema_cutover_keeps_historical_profile_truth():
    for key, profile in CURRENT_OBSERVATION_PROFILES.items():
        assert profile.schema_version == (4 if key[1] == "changelog" else 5)
        legacy = LEGACY_OBSERVATION_PROFILES[key]
        assert legacy.schema_version == 1
        schema = CANONICAL_RECORD_SCHEMAS[(legacy.schema_name, legacy.schema_version)]
        assert any(field.nested_profile == "markdown-structural" for field in schema.fields)


def test_adapter_owns_history_classification_and_core_field_time():
    fields = {"created": "2026-01-01T12:00:00Z", "updated": "2026-05-01T12:00:00Z"}
    comment_event = {"created": "2026-03-01T12:00:00Z", "items": [{"field": "Attachment"}]}
    core_event = {"created": "2026-02-01T12:00:00Z", "items": [{"fieldId": "description"}]}
    assert core_revised_at(fields, [comment_event, core_event], changelog_complete=True) == "2026-02-01T12:00:00+00:00"
    assert core_revised_at(fields, [comment_event], changelog_complete=True) == "2026-01-01T12:00:00+00:00"
    assert core_revised_at(fields, [core_event], changelog_complete=False) is None
    assert changelog_semantic_class(comment_event) == "attachment_event"
    assert changelog_semantic_class({"items": [{"field": "Due Date"}, {"field": "status"}]}) == "operational_transition"
    assert changelog_semantic_class(core_event) == "domain_transition"


@pytest.mark.parametrize('old_count,new_count', [(1, 2), (2, 1), (2, 0)])
def test_history_parent_population_change_keeps_all_current_value_views(old_count, new_count):
    from memforge.pipeline.revision_assessment import RevisionAssessmentContext
    from memforge.pipeline.projection_context import (
        CommittedSourceUnitSnapshot, ExtractionAuthority,
        plan_projection_evidence_work,
    )
    item = {'field': 'description', 'fromString': 'Previous rule.', 'toString': 'New rule.'}
    def histories(count):
        return [{'id': '9', 'created': '2026-01-01T12:00:00Z', 'items': [item] * count}]
    base = projection('Unchanged.', '<p>Unchanged.</p>', histories=histories(old_count))
    target = projection('Unchanged.', '<p>Unchanged.</p>', histories=histories(new_count), prior=base)
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash='scope')
    changed, removed = context.delta_fragments()
    for value in ('Previous rule.', 'New rule.'):
        assert sum(f.presentation_text.endswith(value) for f in changed) == new_count
        assert sum(f.presentation_text.endswith(value) for f in removed) == (old_count if new_count == 0 else 0)
    plan = plan_projection_evidence_work(target,
        committed_base_snapshot=CommittedSourceUnitSnapshot(base.source_unit_revisions[0], base.observation_revisions),
        reprocess_all_current_observations=False)
    assert isinstance(plan, ExtractionAuthority)
    assert not any(plan.authorizes(f) for f in context.full_fragments
                   if any(f.presentation_text.endswith(value) for value in ('Previous rule.', 'New rule.')))


@pytest.mark.parametrize("html", [
    '<p><ins>Keep the condition.</ins></p>',
    '<ins>Keep the condition.</ins>',
    '<table><tr><th><ins>Condition</ins></th></tr><tr><td>Keep it.</td></tr></table>',
])
def test_jira_underline_is_not_an_insertion_event_and_legacy_format_remains_pinned(html):
    from memforge.source_adapters.jira_html import LEGACY_RENDERED_HTML_FORMAT, RENDERED_HTML_FORMAT

    latest = RENDERED_HTML_FORMAT.parse(html)
    underline = RENDERED_HTML_FORMAT.parse(html.replace('<ins>', '<u>').replace('</ins>', '</u>'))
    assert [(f.kind, f.presentation, f.content_value) for f in latest.fragments] == [(f.kind, f.presentation, f.content_value) for f in underline.fragments]
    assert all('[inserted:' not in fragment.presentation for fragment in latest.fragments)
    legacy = LEGACY_RENDERED_HTML_FORMAT.parse(html)
    assert any('[inserted:' in fragment.presentation for fragment in legacy.fragments)
    assert [(f.start, f.end) for f in latest.fragments] == [(f.start, f.end) for f in legacy.fragments]


def test_jira_format_upgrade_preserves_historical_text_views():
    from memforge.source_adapters.jira import _profile

    projected = projection('+Keep the condition.+', '<p><ins>Keep the condition.</ins></p>')
    revision = projected.observation_revisions[0]
    historical = replace(revision, id='historical-v4', evidence_profile=_profile('jira-issue-core', 4))
    old_index = build_revision_fragment_index(historical)
    assert not old_index.errors
    old_fragment = next(f for f in old_index.fragments if '[inserted: Keep the condition.]' in f.presentation_text)
    verify_text_evidence_view(historical, anchor=old_fragment.anchor,
        raw_content_sha256=old_fragment.raw_content_sha256,
        presentation_sha256=old_fragment.presentation_sha256,
        excerpt=old_fragment.presentation_text, text_view=old_fragment.text_view)
    current_index = build_revision_fragment_index(revision)
    assert not current_index.errors
    assert all('[inserted:' not in f.presentation_text for f in current_index.fragments)
    old_unit = replace(projected.source_unit_revisions[0], id='historical-unit-v4', observation_revision_ids=(historical.id,))
    old = replace(projected, observation_revisions=(historical,), source_unit_revisions=(old_unit,), deltas=())
    current = projection('+Keep the condition.+', '<p><ins>Keep the condition.</ins></p>', prior=old)
    _, support = planned(old, current, old_fragment)
    assert support.parts[0].status is not EvidenceCorrespondence.EXACT_UNCHANGED

    current_fragment = next(f for f in build_revision_fragment_index(current.observation_revisions[0]).fragments if f.presentation_text.endswith('Keep the condition.'))
    next_revision = projection('+Keep the condition.+ Add B.', '<p><ins>Keep the condition.</ins></p><p>Add B.</p>', prior=current)
    _, unchanged_support = planned(current, next_revision, current_fragment)
    assert unchanged_support.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED


@pytest.mark.parametrize("tag", ['del', 's', 'strike'])
def test_current_jira_strikethrough_preserves_authored_text_without_a_deletion_event(tag):
    from memforge.source_adapters.jira_html import LEGACY_RENDERED_HTML_FORMAT, RENDERED_HTML_FORMAT

    html = f'<p><{tag}>Old condition.</{tag}> Current condition.</p>'
    latest = RENDERED_HTML_FORMAT.parse(html).fragments[0]
    assert latest.presentation == '[struck through: Old condition.] Current condition.'
    assert '[deleted:' not in latest.presentation
    assert latest.content_value != RENDERED_HTML_FORMAT.parse('<p>Old condition. Current condition.</p>').fragments[0].content_value
    assert LEGACY_RENDERED_HTML_FORMAT.parse(html).fragments[0].presentation == '[deleted: Old condition.] Current condition.'
    assert (latest.start, latest.end) == (0, len(html))
