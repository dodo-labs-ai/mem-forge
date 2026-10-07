"""Native format contracts through parsing, projection and revision planning."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from memforge.pipeline.evidence_fragments import build_revision_fragment_index
from memforge.pipeline.projection_context import CommittedSourceUnitSnapshot, ExtractionAuthority, plan_projection_evidence_work
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.support_reading import EvidenceCorrespondence
from memforge.source_adapters.confluence_storage import parse_storage
from memforge.source_adapters.jira_html import parse_rendered_html
from tests.test_confluence_native_evidence import planned, projection as confluence_projection
from tests.test_jira_native_evidence import projection as jira_projection


FROZEN = json.loads((Path(__file__).parent / "fixtures/native_text_v1_contract.json").read_text())["cases"]
FROZEN_EVIDENCE = json.loads((Path(__file__).parent / "fixtures/native_text_v1_evidence_rows.json").read_text())["cases"]


@pytest.mark.parametrize("case", FROZEN, ids=lambda case: case["source"])
def test_registered_v1_native_contract_preserves_frozen_outputs(case):
    parser = parse_storage if case["source"] == "confluence" else parse_rendered_html
    assert json.loads(json.dumps(asdict(parser(case["input"])))) == case["parsed"]


@pytest.mark.parametrize("case", FROZEN_EVIDENCE, ids=lambda case: case["source"])
def test_persisted_v1_evidence_is_readable_under_compatible_extension(case):
    from memforge.storage.database import Database
    row = case['row']
    result = Database._project_memory_evidence_item(row)
    assert result.interpretation_available
    assert result.excerpt == row['excerpt']
    assert result.anchor.range_start == row['range_start']
    assert result.raw_content_sha256 == row['raw_content_sha256']


@pytest.mark.parametrize('profile', ['markdown', 'plain'])
def test_standard_text_duplicate_growth_authorizes_only_unique_material(profile):
    from tests.test_evidence_fragments import _revision, MARKDOWN_PROFILE, PLAIN_PROFILE
    from memforge.pipeline.evidence_fragments import revision_changed_structural_ranges
    fmt = MARKDOWN_PROFILE if profile == 'markdown' else PLAIN_PROFILE
    base = _revision('TBD\n\nRule A.\n', fmt)
    target = _revision('TBD\n\nRule A.\n\nTBD\n\nRule B.\n', fmt)
    changed = revision_changed_structural_ranges(base, target)
    assert [target.content[start:end].strip() for start, end in changed] == ['Rule B.']
    current = revision_changed_structural_ranges(base, target, purpose='current')
    assert [target.content[start:end].strip() for start, end in current].count('TBD') == 2


def test_layout_cell_heading_does_not_govern_another_column():
    body = '<ac:layout><ac:layout-section ac:type="two_equal"><ac:layout-cell><h2>US only</h2><p>Rule A.</p></ac:layout-cell><ac:layout-cell><p>Rule B.</p></ac:layout-cell></ac:layout-section></ac:layout>'
    parsed = parse_storage(body)
    rule_b = next(f for f in parsed.fragments if f.presentation == 'Rule B.')
    heading = next(f for f in parsed.fragments if f.presentation == 'US only')
    scoped = next(g for g in parsed.groups if g.start == heading.start)
    assert scoped.end <= rule_b.start


@pytest.mark.parametrize("wrapper", [
    '<div class="table-wrap">{body}</div>',
    '<div class="panel" style="border-width: 2px;"><div class="panelContent">{body}</div></div>',
])
def test_jira_layout_keeps_row_granularity_and_interpretation(wrapper):
    table = '<table class="confluenceTable"><tbody><tr><th class="confluenceTh">Rule</th></tr><tr><td class="confluenceTd">Approval required.</td></tr></tbody></table>'
    target = jira_projection('Approval required.', wrapper.format(body=table))
    index = build_revision_fragment_index(target.observation_revisions[0])
    assert not index.errors
    row = next(f for f in index.fragments if f.fragment_type.endswith('table-row'))
    assert row.presentation_text.endswith('Rule: Approval required.')
    assert row.text_view.origins
    assert '<table' not in row.presentation_text


@pytest.mark.parametrize("html,expected", [
    ('<p><a class="user-hover" href="/secure/ViewProfile.jspa?name=alex">Alex</a> approves.</p>', 'Alex (/secure/ViewProfile.jspa?name=alex)'),
    ('<p><a class="user-hover" rel="alex" href="/secure/ViewProfile.jspa?name=alex">Alex</a> approves.</p>', 'user: alex'),
    ('<p><font color="#ff0000">Approval required.</font></p>', '#ff0000'),
    ('<p>Ready <img class="emoticon" src="/images/icons/emoticons/smile.png" alt=":)" /></p>', ':)'),
    ('<div class="preformatted panel" style="border-width: 1px;"><div class="preformattedContent panelContent"><pre>a\n  b</pre></div></div>', 'a\n  b'),
])
def test_jira_semantic_controls_are_readable(html, expected):
    parsed = parse_rendered_html(html)
    assert expected in '\n'.join(f.presentation for f in parsed.fragments)


def test_jira_comment_rendering_preserves_valid_issue_projection():
    target = jira_projection('Core rule.', '<p>Core rule.</p>', comments=[{
        'id': '501', 'body': 'Alex approves.',
        'renderedBody': '<p><a class="user-hover" href="/secure/ViewProfile.jspa?name=alex">Alex</a> approves.</p>',
    }])
    assert len(target.observation_revisions) == 2


@pytest.mark.parametrize("wrap", [
    '<ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>{body}</ac:layout-cell></ac:layout-section></ac:layout>',
    '<ac:structured-macro ac:name="excerpt"><ac:rich-text-body>{body}</ac:rich-text-body></ac:structured-macro>',
])
def test_confluence_container_preserves_evidence_identity(wrap):
    body = '<p>Approval required.</p><p>Audit trail required.</p>'
    base = confluence_projection(body)
    target = confluence_projection(wrap.format(body=body), base)
    old = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments if f.presentation_text == 'Approval required.')
    _, plan = planned(base, target, old)
    assert plan.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert plan.parts[0].current[0].anchor != old.anchor
    assert plan.parts[0].current[0].presentation_text == old.presentation_text


def test_confluence_inline_annotation_preserves_authored_text_and_match():
    base = confluence_projection('<p>Rule A.</p><p>Rule B.</p>')
    target = confluence_projection('<p>Rule <ac:inline-comment-marker ac:ref="x">A</ac:inline-comment-marker>.</p><p>Rule B.</p>', base)
    old = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments if f.presentation_text == 'Rule A.')
    _, plan = planned(base, target, old)
    assert plan.parts[0].status is EvidenceCorrespondence.EXACT_UNCHANGED
    assert len(parse_storage(json.loads(target.observation_revisions[0].content)['body']).fragments) == 2


def test_confluence_task_status_is_meaning_not_layout():
    body = '<ac:task-list><ac:task><ac:task-id>1</ac:task-id><ac:task-status>incomplete</ac:task-status><ac:task-body>Approve rollout.</ac:task-body></ac:task></ac:task-list>'
    before = parse_storage(body).fragments[0]
    after = parse_storage(body.replace('incomplete', 'complete')).fragments[0]
    assert 'incomplete' in before.presentation
    assert 'Approve rollout.' in before.presentation
    assert before.content_value != after.content_value


@pytest.mark.parametrize('name', ['include', 'children', 'excerpt-include', 'unknown-query'])
def test_external_or_unknown_confluence_macro_cannot_fake_complete_evidence(name):
    with pytest.raises(ValueError):
        parse_storage(f'<ac:structured-macro ac:name="{name}" />')


@pytest.mark.parametrize('source', ['jira', 'confluence'])
def test_duplicate_population_does_not_block_unique_new_work(source):
    body = '<p>TBD</p><p>Rule A.</p>'
    target_body = body + '<p>TBD</p><p>Rule B.</p>'
    base = jira_projection('TBD / Rule A.', body) if source == 'jira' else confluence_projection(body)
    target = jira_projection('TBD / Rule A. / TBD / Rule B.', target_body, prior=base) if source == 'jira' else confluence_projection(target_body, base)
    work = plan_projection_evidence_work(target, committed_base_snapshot=CommittedSourceUnitSnapshot(base.source_unit_revisions[0], base.observation_revisions), reprocess_all_current_observations=False)
    assert isinstance(work, ExtractionAuthority)
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash='scope')
    unique = next(f for f in context.full_fragments if f.presentation_text.endswith('Rule B.'))
    assert work.authorizes(unique)
    copies = [f for f in context.full_fragments if f.presentation_text.endswith('TBD')]
    assert len(copies) == 2
    assert all(not work.authorizes(f) for f in copies)
    assert all(f in context.delta_fragments()[0] for f in copies)
