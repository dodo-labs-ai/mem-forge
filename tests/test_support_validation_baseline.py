from dataclasses import replace

import pytest

from tests.test_support_scope_v2 import db as db, _seed_complete_legacy_support


@pytest.mark.asyncio
async def test_reused_evidence_preserves_creation_run(db):
    _, unit_id, _, _ = await _seed_complete_legacy_support(db)
    unit = await db.get_evidence_unit(unit_id)
    await db.db.execute(
        "UPDATE evidence_units SET extractor_run_id = 'run-v1' WHERE id = ?", (unit_id,)
    )
    await db.db.commit()
    await db.upsert_evidence_unit(replace(unit, extractor_run_id="run-v2"))
    stored = await db.get_evidence_unit(unit_id)
    assert stored.extractor_run_id == "run-v1"
    assert stored.doc_revision_id == "unitrev-1"


@pytest.mark.asyncio
async def test_historical_projection_uses_manifest_not_current_observation_pointer(db):
    await _seed_complete_legacy_support(db)
    await db.db.execute(
        "UPDATE source_observations SET current_revision_id = NULL WHERE id = 'obs-primary'"
    )
    await db.db.commit()
    projection = await db.get_source_unit_revision_projection("unit-1", "unitrev-1")
    assert {r.id for r in projection.observation_revisions} == {"obsrev-primary", "obsrev-required"}
    with pytest.raises(ValueError, match="manifest is incomplete"):
        await db.get_current_source_unit_projection("unit-1")
    assert await db.get_source_unit_revision_projection("wrong-unit", "unitrev-1") is None


@pytest.mark.asyncio
async def test_schema_upgrade_leaves_legacy_validation_unknown(db):
    await _seed_complete_legacy_support(db)
    report = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(expected_report_id=report.id, owner_id="test")
    rows = await db.db.execute_fetchall("SELECT validation_plan_id FROM memory_unit_support_assertions")
    assert rows and all(row["validation_plan_id"] is None for row in rows)


@pytest.mark.asyncio
async def test_existing_database_upgrade_preserves_support_and_evidence(db):
    import sqlite3

    await _seed_complete_legacy_support(db)
    report = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(expected_report_id=report.id, owner_id="test-upgrade")
    before = [tuple(row) for row in await db.db.execute_fetchall("SELECT * FROM evidence_units")]
    await db.close()
    with sqlite3.connect(db.db_path) as connection:
        connection.execute("ALTER TABLE memory_unit_support_assertions DROP COLUMN validation_plan_id")
        connection.execute("DELETE FROM schema_migrations WHERE version = 93")
    await db.connect()
    assert [tuple(row) for row in await db.db.execute_fetchall("SELECT * FROM evidence_units")] == before
    [support] = await db.db.execute_fetchall("SELECT active, validation_plan_id FROM memory_unit_support_assertions")
    assert support["active"] == 1 and support["validation_plan_id"] is None
    await db.close()
    await db.connect()
    assert len(await db.db.execute_fetchall("SELECT * FROM memory_unit_support_assertions")) == 1


async def _support_plan(db, memory_id, unit_id, *, plan_id, action):
    from memforge.memory.lifecycle_plan import ReconciliationScope
    from memforge.memory.lifecycle_planner import NewMemoryDefaults, build_lifecycle_plan
    from memforge.memory.evidence import SupportScopeVersion
    from memforge.models import RawMemory, ReconcileAction, ReconcileOperation, content_hash

    memory = await db.get_memory(memory_id)
    unit = await db.get_evidence_unit(unit_id)
    state = (await db.get_active_memory_support_states((memory_id,)))[memory_id]
    return build_lifecycle_plan(
        plan_id=plan_id,
        scope=ReconciliationScope(
            id=plan_id, source_id=unit.source_id, source_unit_id=unit.source_lineage_id,
            base_unit_revision_id=unit.doc_revision_id, target_unit_revision_id=unit.doc_revision_id,
        ),
        gate_state=(await db.get_lifecycle_gate(unit.source_id)).state,
        operations=(ReconcileOperation(
            action=action, memory_id=memory_id,
            memory=RawMemory(content=memory.content, memory_type=memory.memory_type)
                if action is ReconcileAction.NOOP else None,
            reason="test validation" if action is ReconcileAction.NOOP else "source withdrew claim",
        ),),
        incumbents={memory_id: memory}, source_support_reference_ids={}, all_active_support_reference_ids={},
        support_set_hashes={memory_id: state.support_set_hash},
        observation_revision_ids=("obsrev-primary", "obsrev-required"), new_evidence_reference_ids=(),
        support_scope_version=SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
        source_support_unit_ids={memory_id: (unit_id,)}, all_active_support_unit_ids={memory_id: (unit_id,)},
        evidence_unit_ids_by_claim_hash={content_hash(memory.content): (unit_id,)},
        defaults=NewMemoryDefaults(
            visibility="workspace", owner_user_id=None, project_key=None, repo_identifier=None,
            doc_id=unit.doc_id, source_type=unit.source_type, access_context_hash=unit.access_context_hash,
        ),
    )


@pytest.mark.asyncio
async def test_gated_review_retains_last_successful_support_plan(db):
    from memforge.models import ReconcileAction

    memory_id, unit_id, source_id, _ = await _seed_complete_legacy_support(db)
    report = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(expected_report_id=report.id, owner_id="test-review")
    await db.enable_lifecycle_gate(source_id)
    for plan_id, action in (("verified-plan", ReconcileAction.NOOP), ("review-plan", ReconcileAction.DELETE)):
        if action is ReconcileAction.DELETE:
            await db.gate_destructive_lifecycle(source_id, reason="unresolved historical provenance")
        plan = await _support_plan(db, memory_id, unit_id, plan_id=plan_id, action=action)
        await db.apply_lifecycle_plan(plan)
        support = await db.get_active_memory_support_evidence(memory_id, source_id=source_id)
        assert {item.validation_plan_id for item in support} == {"verified-plan"}
        assert {item.validation_unit_revision_id for item in support} == {"unitrev-1"}
    assert len(await db.list_lifecycle_reviews(source_id)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("other_has_baseline", [False, True])
async def test_shared_evidence_keeps_each_memory_validation_baseline_independent(db, other_has_baseline):
    from memforge.memory.evidence import MemoryUnitSupportAssertion, memory_unit_support_assertion_id
    from memforge.models import MemorySource, ReconcileAction, content_hash

    memory_id, unit_id, source_id, access_hash = await _seed_complete_legacy_support(db)
    report = await db.report_support_scope_cutover()
    await db.apply_support_scope_v2_cutover(expected_report_id=report.id, owner_id="test-shared")
    await db.enable_lifecycle_gate(source_id)
    memory = await db.get_memory(memory_id)
    unit = await db.get_evidence_unit(unit_id)
    other = replace(memory, id="mem-other", content="Other fixed claim", content_hash=content_hash("Other fixed claim"))
    await db.insert_memory(other)
    await db.restore_memory_source_snapshot(MemorySource(
        memory_id=other.id, doc_id=unit.doc_id, source_id=source_id,
        source_type=unit.source_type, excerpt=unit.excerpt, source_updated_at=None,
    ))
    await db.upsert_memory_unit_support_assertion(MemoryUnitSupportAssertion(
        id=memory_unit_support_assertion_id(
            memory_id=other.id, evidence_unit_id=unit_id, source_id=source_id, access_context_hash=access_hash,
        ),
        memory_id=other.id, evidence_unit_id=unit_id, source_id=source_id, access_context_hash=access_hash,
    ))
    if other_has_baseline:
        await db.apply_lifecycle_plan(await _support_plan(
            db, other.id, unit_id, plan_id="other-verified-plan", action=ReconcileAction.NOOP,
        ))
    before = await db.get_active_memory_support_evidence(other.id, source_id=source_id)
    assert before
    assert {item.validation_plan_id for item in before} == {"other-verified-plan" if other_has_baseline else None}
    assert {item.validation_unit_revision_id for item in before} == {"unitrev-1" if other_has_baseline else None}
    stored_unit = await db.get_evidence_unit(unit_id)

    await db.apply_lifecycle_plan(await _support_plan(
        db, memory_id, unit_id, plan_id="first-verified-plan", action=ReconcileAction.NOOP,
    ))
    supports = await db.get_active_memory_support_evidence_many((memory_id, other.id), source_id=source_id)
    assert {item.evidence_unit_id for items in supports.values() for item in items} == {unit_id}
    assert {item.validation_plan_id for item in supports[memory_id]} == {"first-verified-plan"}
    assert {item.validation_unit_revision_id for item in supports[memory_id]} == {"unitrev-1"}
    assert supports[other.id] == before
    assert await db.get_evidence_unit(unit_id) == stored_unit
