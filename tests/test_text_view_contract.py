"""Public binding/selection seams for source-bound readable interpretation."""

import hashlib
from dataclasses import replace

import pytest

from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole
from memforge.memory.text_view import TextEvidenceView
from memforge.pipeline.evidence_fragments import build_revision_fragment_index, verify_text_evidence_view
from memforge.pipeline.projection_context import CommittedSourceUnitSnapshot, ExtractionAuthority, plan_projection_evidence_work
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.support_reading import EvidenceCorrespondence, correspond_evidence
from tests.test_confluence_native_evidence import projection


def authority(base, target):
    result = plan_projection_evidence_work(target,
        committed_base_snapshot=CommittedSourceUnitSnapshot(base.source_unit_revisions[0], base.observation_revisions),
        reprocess_all_current_observations=False)
    assert isinstance(result, ExtractionAuthority)
    return result


def test_native_insertion_authorizes_only_added_content_and_matches_all_unique_refs():
    native = '<h2>Eligibility</h2><p>Require explicit approval.</p><table><tr><th>Condition</th><th>Outcome</th></tr><tr><td>After cutoff</td><td>Reject assignment.</td></tr></table>'
    base = projection(native)
    target = projection('<p>New independently authored assertion.</p>' + native, base)
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash='scope')
    eligible = [f for f in context.full_fragments if authority(base, target).authorizes(f)]
    assert [f.presentation_text for f in eligible] == ['New independently authored assertion.']
    old = build_revision_fragment_index(base.observation_revisions[0]).fragments
    parts = tuple(ActiveSupportEvidence(memory_id='m', source_id=base.source_id, reference_id=str(i),
        evidence_unit_id='eu', role=EvidenceRole.PRIMARY, anchor=f.anchor, excerpt=f.presentation_text,
        raw_content_sha256=f.raw_content_sha256, presentation_sha256=f.presentation_sha256,
        text_view=f.text_view) for i, f in enumerate(old))
    assert all(c.status is EvidenceCorrespondence.EXACT_UNCHANGED for c in correspond_evidence(context, parts))


def test_header_and_inherited_cell_have_exact_origins_and_cannot_be_fabricated():
    native = '<table><tr><th>Region</th><th>Rule</th></tr><tr><td rowspan="2">US</td><td>Require approval.</td></tr><tr><td>Require two reviewers.</td></tr></table>'
    revision = projection(native).observation_revisions[0]
    index = build_revision_fragment_index(revision)
    row = next(f for f in index.fragments if 'Require two reviewers.' in f.presentation_text)
    view = row.text_view
    assert view is not None and len(view.origins) == 2
    assert TextEvidenceView.from_payload(view.payload()) == view
    for origin in view.origins:
        assert hashlib.sha256(revision.content[origin.anchor.range_start:origin.anchor.range_end].encode()).hexdigest() == origin.raw_content_sha256
    kwargs = dict(anchor=row.anchor, raw_content_sha256=row.raw_content_sha256,
                  presentation_sha256=row.presentation_sha256, excerpt=row.presentation_text, index=index)
    verify_text_evidence_view(revision, text_view=view, **kwargs)
    for bad in (replace(view, origins=()), replace(view, version=99),
                replace(view, origins=(replace(view.origins[0], raw_content_sha256='0' * 64), *view.origins[1:]))):
        with pytest.raises(ValueError):
            verify_text_evidence_view(revision, text_view=bad, **kwargs)


def test_changed_header_authorizes_affected_row_while_core_can_still_correspond():
    native = '<table><tr><th>Condition</th><th>Allowed</th></tr><tr><td>After cutoff</td><td>No</td></tr></table>'
    base = projection(native)
    target = projection(native.replace('Allowed', 'Rejected'), base)
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash='scope')
    row = next(f for f in context.full_fragments if f.fragment_type.endswith('table-row'))
    assert authority(base, target).authorizes(row)
    assert 'Rejected: No' in row.presentation_text


@pytest.mark.parametrize('source_type', ['confluence', 'jira'])
def test_native_list_lead_in_change_authorizes_only_its_governed_list(source_type):
    from tests.test_jira_native_evidence import projection as jira_projection
    native = '<p>US policy applies to:</p><ul><li>Require approval.</li><li>Keep an audit trail.</li></ul><p>Unrelated rule:</p><ul><li>Use two replicas.</li></ul>'
    changed = native.replace('US policy', 'Canada policy')
    base = projection(native) if source_type == 'confluence' else jira_projection(native, native)
    target = projection(changed, base) if source_type == 'confluence' else jira_projection(changed, changed, prior=base)
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash='scope')
    governed = next(f for f in context.full_fragments if 'Require approval.' in f.presentation_text)
    unrelated = next(f for f in context.full_fragments if 'Use two replicas.' in f.presentation_text)
    assert authority(base, target).authorizes(governed)
    assert not authority(base, target).authorizes(unrelated)


def test_extraction_payload_preserves_view_without_persisting_transient_refs():
    from memforge.models import MemoryExtractionResult, RawMemory
    from memforge.source_derivation import memory_extraction_result_from_output_payload, memory_extraction_output_payload
    source = projection('<table><tr><th>Rule</th></tr><tr><td>Require approval.</td></tr></table>')
    context = RevisionAssessmentContext(projection=source, base=None, access_context_hash='scope')
    catalog = context.catalog(context.full_fragments)
    row = next(f for f in catalog.fragments if f.fragment_type.endswith('table-row'))
    selection = catalog.resolve_selection(primary_ref=row.reference, required_refs=[])
    raw = RawMemory(content='Approval is required.', memory_type='fact', resolved_evidence_selection=selection)
    result = MemoryExtractionResult(memories=[raw])
    payload = memory_extraction_output_payload(result)
    restored = memory_extraction_result_from_output_payload(payload)
    assert restored.memories[0].resolved_evidence_selection.parts[0].text_view == row.text_view


def test_native_scope_change_authorizes_governed_core_but_not_another_section():
    native = '<h2>US policy</h2><p>Two approvers are required.</p><h2>Other policy</h2><p>Keep an audit trail.</p>'
    base = projection(native)
    target = projection(native.replace('US policy', 'Canada policy'), base)
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash='scope')
    work = authority(base, target)
    governed = next(f for f in context.full_fragments if f.presentation_text == 'Two approvers are required.')
    unrelated = next(f for f in context.full_fragments if f.presentation_text == 'Keep an audit trail.')
    assert work.authorizes(governed)
    assert not work.authorizes(unrelated)
    old = next(f for f in build_revision_fragment_index(base.observation_revisions[0]).fragments if f.presentation_text == governed.presentation_text)
    part = ActiveSupportEvidence(memory_id='m', source_id=base.source_id, reference_id='r', evidence_unit_id='eu',
        role=EvidenceRole.PRIMARY, anchor=old.anchor, excerpt=old.presentation_text,
        raw_content_sha256=old.raw_content_sha256, presentation_sha256=old.presentation_sha256, text_view=old.text_view)
    assert correspond_evidence(context, (part,))[0].status is EvidenceCorrespondence.EXACT_UNCHANGED


def stored_view_row(fragment, revision):
    import json
    profile = revision.evidence_profile
    return dict(id='ref', role='primary', part_kind='text', observation_id=revision.observation_id,
        observation_revision_id=revision.id, anchor_kind=fragment.anchor.kind.value,
        range_start=fragment.anchor.range_start, range_end=fragment.anchor.range_end, fragment_id=None,
        current_revision_id=revision.id, content=revision.content, metadata_json='{}', observation_type='page_body',
        profile_name=profile.name, profile_version=profile.version, coordinate_space=profile.coordinate_space.value,
        representation_schema_name=profile.schema_name, representation_schema_version=profile.schema_version,
        excerpt=fragment.presentation_text, unit_excerpt=fragment.presentation_text, display_text=None,
        raw_content_sha256=fragment.raw_content_sha256, presentation_sha256=fragment.presentation_sha256,
        text_view_json=json.dumps(fragment.text_view.payload()))


def test_sqlite_delivery_preserves_view_and_history_but_rejects_corrupt_origins():
    import json
    from memforge.storage.database import Database
    revision = projection('<table><tr><th>Rule</th></tr><tr><td>Require approval.</td></tr></table>').observation_revisions[0]
    fragment = next(f for f in build_revision_fragment_index(revision).fragments if f.fragment_type.endswith('table-row'))
    row = stored_view_row(fragment, revision)
    delivered = Database._project_memory_evidence_item(row)
    assert delivered.text_view == fragment.text_view and delivered.interpretation_available
    descriptor = json.loads(row['text_view_json'])
    descriptor['version'] = 99
    historical = Database._project_memory_evidence_item({**row, 'text_view_json': json.dumps(descriptor)})
    assert historical.reference_id == delivered.reference_id and historical.excerpt == delivered.excerpt
    assert not historical.interpretation_available
    descriptor['origins'][0]['raw_content_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='origin integrity'):
        Database._project_memory_evidence_item({**row, 'text_view_json': json.dumps(descriptor)})


@pytest.mark.parametrize('source_type', ['confluence', 'jira', 'markdown-structural', 'plain-text'])
@pytest.mark.parametrize('old_count,new_count', [(1, 2), (2, 1), (2, 0)])
def test_duplicate_population_reads_all_current_without_inventing_removed_occurrence(source_type, old_count, new_count):
    from tests.test_jira_native_evidence import projection as jira_projection
    from memforge.source_representation import MARKDOWN_STRUCTURAL_PROFILE, PLAIN_TEXT_PROFILE

    def make(count, prior=None):
        native = '<p>Same rule.</p>' * count
        if source_type == 'jira':
            return jira_projection(native, native, prior=prior)
        value = projection(native, prior)
        if source_type in {'markdown-structural', 'plain-text'}:
            revision = value.observation_revisions[0]
            body = '\n\n'.join(['Same rule.'] * count)
            profile = MARKDOWN_STRUCTURAL_PROFILE if source_type == 'markdown-structural' else PLAIN_TEXT_PROFILE
            value = replace(value, observation_revisions=(replace(revision, content=body,
                semantic_hash=hashlib.sha256(body.encode()).hexdigest(), evidence_profile=profile),))
        return value

    base = make(old_count)
    target = make(new_count, base)
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash='scope')
    changed, removed = context.delta_fragments()
    def relevant(fragments):
        return [f for f in fragments if f.presentation_text.endswith('Same rule.')]
    assert len(relevant(changed)) == new_count
    assert len(relevant(removed)) == (old_count if new_count == 0 else 0)
    plan = plan_projection_evidence_work(target,
        committed_base_snapshot=CommittedSourceUnitSnapshot(base.source_unit_revisions[0], base.observation_revisions),
        reprocess_all_current_observations=False)
    assert isinstance(plan, ExtractionAuthority)
    assert not any(plan.authorizes(f) for f in relevant(context.full_fragments))
    if new_count:
        old_fragments = relevant(build_revision_fragment_index(base.observation_revisions[0]).fragments)
        parts = tuple(ActiveSupportEvidence(memory_id='m', source_id=base.source_id,
            reference_id=str(i), evidence_unit_id='eu', role=EvidenceRole.PRIMARY,
            anchor=f.anchor, excerpt=f.presentation_text, raw_content_sha256=f.raw_content_sha256,
            presentation_sha256=f.presentation_sha256, text_view=f.text_view)
            for i, f in enumerate(old_fragments))
        assert all(result.status is EvidenceCorrespondence.AMBIGUOUS
                   for result in correspond_evidence(context, parts))
