"""Focused presentation cannot replace source authority or hide Required changes."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from memforge.llm.structured import ProjectionFragmentMemoryCandidate
from memforge.memory.evidence import EvidenceReference, evidence_part_set_digest
from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole
from memforge.models import Memory, content_hash
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.support_reading import SupportWorkItem, plan_support_revision, exact_evidence_selection
from memforge.pipeline.memory_extractor import _SelectionResolution
from memforge.pipeline.revision_work import RevisionWorkExecutor
from memforge.plugin_mcp_proxy import _compact_memory_response
from memforge.server.admin_api import _memory_evidence_unit_detail
from memforge.source_derivation import memory_extraction_output_payload, memory_extraction_result_from_output_payload
from memforge.models import MemoryExtractionResult
from memforge.storage.database import Database
from tests.test_evidence_unit_support import _seed_complete_unit_support
from tests.test_projection_fragments import _projection, _compile
from tests.test_revision_work import RULE, work_items
from tests.unit_support_fixture import record_unit_support
from tests.test_confluence_native_evidence import projection


@pytest.mark.parametrize("displays", [[], [{"ref": "unknown", "text": "A rule."}],
    [{"ref": "p1", "text": "A rule."}, {"ref": "p1", "text": "Duplicate."}]])
def test_missing_extra_or_duplicate_display_is_a_technical_failure(displays):
    with pytest.raises(ValidationError, match="evidence_displays"):
        ProjectionFragmentMemoryCandidate(content="A rule.", memory_type="fact", primary_ref="p1",
                                          evidence_displays=displays)


def test_required_order_is_resolved_by_identity_and_display_does_not_change_authority():
    source = _projection(primary_content="Intro.\n\nApproval is required.\n\nKeep audit records.\n")
    catalog = _compile(source, access_context_hash="scope")
    primary = next(f.reference for f in catalog.fragments if f.primary_eligible)
    required = [f.reference for f in catalog.fragments if f.reference != primary][:2]
    assert len(required) == 2
    candidate = ProjectionFragmentMemoryCandidate(content="Approval is required.", memory_type="fact",
        primary_ref=primary, required_refs=required[::-1],
        evidence_displays=[{"ref": r, "text": "Readable " + r} for r in [primary, *required]])
    [raw] = _SelectionResolution().resolve([candidate], catalog=catalog, prompt_hash="0" * 64, returned=1)
    bound = raw.resolved_evidence_selection
    by_anchor = {f.anchor: f.reference for f in catalog.fragments}
    assert all(p.display_text == "Readable " + by_anchor[p.anchor] for p in bound.parts)
    original = catalog.resolve_selection(primary_ref=primary, required_refs=required)
    assert [(p.anchor, p.raw_content_sha256, p.presentation_sha256, p.excerpt, p.text_view)
            for p in bound.parts] == [(p.anchor, p.raw_content_sha256, p.presentation_sha256, p.excerpt, p.text_view)
                                     for p in original.parts]
    payload = memory_extraction_output_payload(MemoryExtractionResult(memories=[raw]))
    restored = memory_extraction_result_from_output_payload(payload)
    assert restored.memories[0].resolved_evidence_selection == bound


@pytest.mark.asyncio
async def test_program_rebind_preserves_display_only_for_unchanged_complete_presentation():
    [item] = work_items(RULE)
    item = replace(item, support=tuple(replace(p, display_text="Two reviewers approve US releases.")
                                      for p in item.support))
    executor = RevisionWorkExecutor(client=None, model="unused")
    result = (await executor.assess_many([item]))[item.id]
    assert result.supported and result.rebound and executor.calls == 0
    assert result.memory.resolved_evidence_selection.parts[0].display_text == "Two reviewers approve US releases."


def test_changed_interpretation_never_carries_old_display_to_the_new_revision():
    native = '<table><tr><th>Rule</th></tr><tr><td>Require approval.</td></tr></table>'
    base = projection(native)
    target = projection(native.replace('Rule', 'Instruction'), base)
    old_context = RevisionAssessmentContext(projection=base, base=None, access_context_hash='scope')
    row = next(f for f in old_context.full_fragments if f.fragment_type.endswith('table-row'))
    part = ActiveSupportEvidence(memory_id='memory', source_id=base.source_id, reference_id='ref',
        evidence_unit_id='unit', role=EvidenceRole.PRIMARY, anchor=row.anchor, excerpt=row.presentation_text,
        raw_content_sha256=row.raw_content_sha256, presentation_sha256=row.presentation_sha256,
        text_view=row.text_view, display_text='Rule: Require approval.')
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash='scope')
    memory = Memory(id='memory', content='Approval is required.', memory_type='fact',
                    content_hash=content_hash('Approval is required.'))
    item = SupportWorkItem('work', memory, (part,), context)
    plan = plan_support_revision(context, [item])
    rebound = RevisionWorkExecutor(client=None, model='unused')._rebound(plan.catalog, plan.supports[0])
    selected = rebound.memory.resolved_evidence_selection.parts[0]
    assert selected.display_text is None and 'Instruction' in selected.excerpt
    assert exact_evidence_selection(context, (part,)).parts[0].display_text is None


@pytest.mark.asyncio
async def test_assessment_generates_current_display_inside_the_existing_model_response():
    from tests.test_revision_work import Client, CHANGED
    [item] = work_items(CHANGED)
    class CurrentDisplay(Client):
        def judge(self, prompt):
            rows = super().judge(prompt)
            for row in rows:
                if row['status'] == 'supported':
                    row['evidence_displays'] = [{'ref': row['primary_ref'], 'text': RULE}]
            return rows
    client = CurrentDisplay()
    result = (await RevisionWorkExecutor(client=client, model='fixture').assess_many([item]))[item.id]
    assert result.supported and result.memory.resolved_evidence_selection.parts[0].display_text == RULE
    assert len(client.prompts) == 1 and len(client.impact_prompts) == 1


@pytest.mark.asyncio
async def test_sqlite_roundtrip_tool_delivery_and_original_evidence_remain_separate(tmp_path):
    database_path = tmp_path / "display.db"
    db = Database(str(database_path))
    await db.connect()
    try:
        memory_id, unit_id, source_id, _ = await _seed_complete_unit_support(db)
        unit = await db.get_evidence_unit(unit_id)
        group = next(g for g in await db.get_memory_evidence_units(memory_id) if g.evidence_unit_id == unit_id)
        refs = tuple(EvidenceReference(role=p.role, kind=p.kind, anchor=p.anchor,
            raw_content_sha256=p.raw_content_sha256, presentation_sha256=p.presentation_sha256,
            excerpt=p.excerpt, text_view=p.text_view) for p in group.items)
        focused = tuple(replace(r, id=None, evidence_unit_id=None,
                                display_text="Focused " + r.role.value) for r in refs)
        assert evidence_part_set_digest(refs) == evidence_part_set_digest(focused)
        new_unit = replace(unit, id="focused-unit")
        await record_unit_support(db, memory_id=memory_id, unit=new_unit, references=focused)
    finally:
        await db.close()
    db = Database(str(database_path))
    await db.connect()
    try:
        groups = await db.get_memory_evidence_units(memory_id)
        group = next(g for g in groups if g.evidence_unit_id == "focused-unit")
        assert [p.display_text for p in group.items] == ["Focused primary", "Focused required"]
        detail = _memory_evidence_unit_detail(group, document=None, memory_id=memory_id).model_dump(mode="json")
        compact = _compact_memory_response({"id": memory_id, "content": "Claim.", "evidence": [detail]})
        assert [p["text"] for p in compact["evidence"][0]["items"]] == ["Focused primary", "Focused required"]
        assert all("excerpt" not in p and p["resource_url"] for p in compact["evidence"][0]["items"])
        assert all(p["excerpt"] and p["text"] != p["excerpt"] for p in detail["items"])
        active = (await db.get_active_memory_support_evidence_many((memory_id,), source_id=source_id))[memory_id]
        assert [p.display_text for p in active if p.evidence_unit_id == "focused-unit"] == ["Focused primary", "Focused required"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sqlite_upgrade_keeps_existing_refs_without_inventing_display(tmp_path):
    path = tmp_path / "before-display.db"
    db = Database(str(path))
    await db.connect()
    try:
        memory_id, unit_id, _, _ = await _seed_complete_unit_support(db)
        before = next(g for g in await db.get_memory_evidence_units(memory_id)
                      if g.evidence_unit_id == unit_id).items
        # Reconstruct the previous additive schema with actual stored Evidence.
        await db.db.execute("ALTER TABLE evidence_references DROP COLUMN display_text")
        await db.db.execute("DELETE FROM schema_migrations WHERE version = ?", (110,))
        await db.db.commit()
    finally:
        await db.close()
    upgraded = Database(str(path))
    await upgraded.connect()
    try:
        groups = await upgraded.get_memory_evidence_units(memory_id)
        group = next(g for g in groups if g.evidence_unit_id == unit_id)
        assert group.items == before and all(ref.display_text is None for ref in group.items)
        detail = _memory_evidence_unit_detail(group, document=None, memory_id=memory_id)
        assert all(item.text == item.excerpt and item.resource_url for item in detail.items)
    finally:
        await upgraded.close()
