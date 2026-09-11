from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.derivation_work import DerivationWork, payload_hash
from memforge.models import DocumentRecord, ContentItem, RawContent, NormalizedContent
from memforge.source_derivation import SourceUnitDerivationContext, source_derivation_manifest
from memforge.storage.database import Database
from memforge.pipeline.source_projection_adapters import project_source_item


def staged_fixture(projection=None):
    now = datetime(2026, 9, 7, tzinfo=timezone.utc)
    if projection is None:
        body = "Two reviewers approve US releases.\n"
        item = ContentItem(item_id="doc", title="Stage fixture", source_url="https://example.test/doc",
                           last_modified=now, content_type="text/markdown", version="1")
        projection = project_source_item(source_id="source-1", source_type="github_repo", run_id="run-1",
            item=item, raw=RawContent(item=item, body=body.encode(), content_type="text/markdown"),
            normalized=NormalizedContent(item=item, markdown_body=body), scope={}, access_context={"visibility": "workspace"})
    document = DocumentRecord(
        doc_id="doc",
        source=projection.source_id,
        source_url="https://example.test/doc",
        title="Stage fixture",
        space_or_project="TEST",
        author=None,
        last_modified=now,
        labels=[],
        version="1",
        content_hash="hash",
        token_count=20,
        raw_content_uri=None,
        raw_content_type="text/markdown",
        normalized_content_uri=None,
        pdf_content_uri=None,
        last_synced=now,
    )
    context = SourceUnitDerivationContext(
        document=document,
        doc_type="document",
        project_key=None,
        repo_identifier=None,
        document_content=projection.observation_revisions[0].content,
        update_mode="new",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=now.isoformat(),
        user_id=None,
        source_activity_epoch=None,
    )
    return projection, source_derivation_manifest(projection, (), context=context)


async def prepare_database(path, projection=None):
    db = Database(str(path))
    await db.connect()
    projection, manifest = staged_fixture(projection)
    await db.upsert_source(
        id=projection.source_id,
        type=projection.source_type,
        name="Stage fixture",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="fixture-user",
    )
    await db.stage_source_derivation(manifest)
    return db, manifest


@pytest.mark.asyncio
async def test_durable_work_reuses_completed_result_after_reopen_and_checks_dependencies(tmp_path):
    path = tmp_path / "stages.db"
    db, root = await prepare_database(path)
    scan = DerivationWork.create(
        "support_scan", {"target": root.target_unit_revision_id, "dependencies": [], "support": [("eu1", "e1", "rev1")]}
    )
    child = DerivationWork.create("support_finalize", {"dependencies": [[scan.id, payload_hash({"results": []})]]})
    try:
        with pytest.raises(ValueError, match="dependency missing"):
            await db.stage_derivation_work(derivation_id=root.id, work=child)
        assert await db.stage_derivation_work(derivation_id=root.id, work=scan) == scan
        with pytest.raises(ValueError, match="dependency incomplete"):
            await db.stage_derivation_work(derivation_id=root.id, work=child)
        result = {"results": []}
        completed = replace(scan, status="completed", result=result, result_hash=payload_hash(result))
        await db.record_derivation_work(derivation_id=root.id, work=completed)
    finally:
        await db.close()
    db = Database(str(path))
    await db.connect()
    try:
        assert await db.stage_derivation_work(derivation_id=root.id, work=scan) == completed
        late_failure = replace(scan, status="retryable_failure", error_code="TimeoutError")
        assert await db.record_derivation_work(derivation_id=root.id, work=late_failure) == completed
        await db.stage_derivation_work(derivation_id=root.id, work=child)
        with pytest.raises(ValueError, match="live root"):
            await db.stage_derivation_work(derivation_id="another-root", work=child)
        changed_model = DerivationWork.create("support_scan", {**scan.manifest, "model": "different-model"})
        assert changed_model.id != scan.id
        assert (await db.stage_derivation_work(derivation_id=root.id, work=changed_model)).status == "pending"
    finally:
        await db.close()


def test_corrupt_completed_result_and_manifest_are_rejected():
    work = DerivationWork.create("support_scan", {})
    with pytest.raises(ValueError, match="result mismatch"):
        DerivationWork.from_payload(
            replace(work, status="completed", result={"x": 1}, result_hash="wrong").to_payload()
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        DerivationWork.from_payload(replace(work, manifest={"kind": "support_scan", "model": "changed"}).to_payload())


@pytest.mark.asyncio
async def test_executor_resumes_from_actual_sqlite_roundtrip(tmp_path):
    from tests.test_revision_work import Client, work_items
    from memforge.pipeline.revision_work import RevisionWorkExecutor

    items = work_items(
        "Two reviewers approve US releases.\n\n" + "\n\n".join(f"Routine operational note {i}." for i in range(160))
    )
    path = tmp_path / "executor.db"
    db, root = await prepare_database(path, items[0].context.projection)
    client = Client()
    client.fail_at = 2
    first = RevisionWorkExecutor(client=client, model="fixture", store=db, derivation_id=root.id)
    try:
        with pytest.raises(TimeoutError):
            await first.assess_many(items)
        [row] = await db.db.execute_fetchall(
            "SELECT COUNT(*) AS n FROM source_derivation_work WHERE derivation_id = ? AND json_extract(payload_json, '$.status') = 'completed'",
            (root.id,),
        )
        completed_count = row["n"]
        assert completed_count > 0
    finally:
        await db.close()
    db = Database(str(path))
    await db.connect()
    try:
        retry = RevisionWorkExecutor(client=client, model="fixture", store=db, derivation_id=root.id)
        assert (await retry.assess_many(items))["w0"].supported
        assert retry.reused == completed_count
        assert retry.final_work_ids
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("work_kind", ["support_finalize", "claim_assess"])
async def test_atomic_commit_requires_complete_final_work(tmp_path, work_kind):
    from memforge.memory.lifecycle_plan import (
        LifecyclePlan,
        ReconciliationScope,
        LifecycleGateState,
        CoverageProof,
        StaleGuard,
    )
    from memforge.source_derivation import source_unit_derivation_context_from_payload

    db, root = await prepare_database(tmp_path / "commit.db")
    projection, _ = staged_fixture()
    delta = projection.deltas[0]
    plan = LifecyclePlan(
        id="plan-stage-check",
        scope=ReconciliationScope(
            id="scope",
            source_id=projection.source_id,
            source_unit_id=delta.source_unit_id,
            base_unit_revision_id=delta.previous_unit_revision_id,
            target_unit_revision_id=delta.current_unit_revision_id,
        ),
        gate_state=LifecycleGateState.GATED,
        coverage_proof=CoverageProof((), (), (), ()),
        stale_guard=StaleGuard((), {}),
        mutations=(),
    )
    final = DerivationWork.create(work_kind, {"dependencies": []})
    kwargs = dict(
        document=source_unit_derivation_context_from_payload(root.context_payload).document,
        derivation_id=root.id,
        derivation_context_identity_hash=root.context_identity_hash,
        required_derivation_work_ids=(final.id,),
    )
    try:
        with pytest.raises(ValueError, match="work missing at commit"):
            await db.apply_source_projection_lifecycle(projection, plan, **kwargs)
        await db.stage_derivation_work(derivation_id=root.id, work=final)
        with pytest.raises(ValueError, match="work incomplete at commit"):
            await db.apply_source_projection_lifecycle(projection, plan, **kwargs)
        assert await db.get_current_source_unit_revision(delta.source_unit_id) is None
        await db.record_derivation_work(
            derivation_id=root.id,
            work=replace(final, status="completed", result={"results": []}, result_hash=payload_hash({"results": []})),
        )
        await db.apply_source_projection_lifecycle(projection, plan, **kwargs)
        assert (await db.get_current_source_unit_revision(delta.source_unit_id)).id == delta.current_unit_revision_id
    finally:
        await db.close()
