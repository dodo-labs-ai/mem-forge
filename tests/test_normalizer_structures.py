from memforge.pipeline.normalizer_utils import html_to_markdown, strip_boilerplate
from memforge.pipeline.evidence_fragments import compile_fragments
from memforge.memory.evidence import EvidenceRole
from tests.test_evidence_fragments import _revision, _authority, MARKDOWN_PROFILE


def test_normalization_retains_html_semantics_before_catalog_compilation():
    content = ('<h1>US payroll</h1><table><tr><th rowspan="2">Scope</th>'
               '<th colspan="2">Approval</th></tr><tr><td>Sandbox</td><td>Small Box</td></tr>'
               '<tr><td>Visibility</td><td>Yes</td><td>No</td></tr></table>'
               '<dl><dt>Cedar</dt><dd>US payroll</dd></dl>'
               '<figure><img src="flow.png"/><figcaption>US approval flow</figcaption></figure>'
               '<ol start="4"><li>Stop</li><li>Update</li></ol>'
               '<pre>if approved:\n    submit()\n\n\n# Table of Contents\n    end()</pre>')
    markdown = strip_boilerplate(html_to_markdown(content))
    revision = _revision(markdown, MARKDOWN_PROFILE)
    catalog = compile_fragments(revision, (_authority(revision, EvidenceRole.PRIMARY),))
    assert not catalog.errors
    by_kind = {f.fragment_type: f.presentation_text for f in catalog.fragments}
    assert 'rowspan="2"' in by_kind['html-table'] and 'colspan="2"' in by_kind['html-table']
    assert 'Sandbox' in by_kind['html-table'] and 'Small Box' in by_kind['html-table']
    assert 'Cedar' in by_kind['html-dl'] and 'US payroll' in by_kind['html-dl']
    assert 'flow.png' in by_kind['html-figure'] and 'US approval flow' in by_kind['html-figure']
    assert 'start="4"' in by_kind['html-ol']
    assert '    submit()\n\n\n# Table of Contents\n    end()' in by_kind['html-pre']


def test_plain_table_remains_compact_markdown_but_one_whole_reference():
    text = html_to_markdown('<table><tr><th>Country</th><th>Approval</th></tr><tr><td>US</td><td>Two</td></tr></table>')
    revision = _revision(text, MARKDOWN_PROFILE)
    [table] = compile_fragments(revision, (_authority(revision, EvidenceRole.PRIMARY),)).fragments
    assert table.fragment_type == 'markdown-table'
    assert 'Country' in table.presentation_text and 'Two' in table.presentation_text


def test_structured_cells_and_caption_keep_complete_table():
    content = ('<table><caption>US only</caption><tr><th>Step</th><th>Rule</th></tr>'
               '<tr><td><ol start="4"><li>Stop</li><li>Start</li></ol></td>'
               '<td><pre>if approved:\n    submit()</pre></td></tr></table>')
    markdown = strip_boilerplate(html_to_markdown(content))
    revision = _revision(markdown, MARKDOWN_PROFILE)
    [table] = compile_fragments(revision, (_authority(revision, EvidenceRole.PRIMARY),)).fragments
    assert table.fragment_type == 'html-table'
    assert 'US only' in table.presentation_text and 'start="4"' in table.presentation_text
    assert 'if approved:\n    submit()' in table.presentation_text


def test_protected_code_at_document_boundary_keeps_indentation():
    code = '    if approved:\n        submit()\n'
    assert strip_boilerplate(code) == code.rstrip('\n')
