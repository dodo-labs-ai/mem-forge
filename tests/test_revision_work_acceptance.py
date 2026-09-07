"""Integration coverage with fixture model judgments; not model-accuracy evidence."""

import json
import pytest
from memforge.llm.structured import SupportAssessmentResponse, SupportAssessmentResult
from memforge.pipeline.revision_work import RevisionWorkExecutor
from memforge.storage.database import Database
from tests.test_derivation_work import prepare_database
from tests.test_revision_work import Client, work_items, payload


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, 3])
async def test_three_range_dependency_and_sqlite_resume(tmp_path, failure):
    pieces = [
        "Two reviewers approve US releases.",
        "Alder releases need not have two reviewers.",
        "Alder refers to Birch.",
        "Birch refers to US releases.",
    ]
    filler = "\n\n".join(f"Routine note {i}: maintain settings." for i in range(140))
    items = work_items(("\n\n" + filler + "\n\n").join(pieces))
    path = tmp_path / "dependencies.db"
    db, root = await prepare_database(path, items[0].context.projection)
    client = Client()
    client.fail_at = failure
    executor = RevisionWorkExecutor(client=client, model="fixture", store=db, derivation_id=root.id)
    try:
        if failure:
            with pytest.raises(TimeoutError):
                await executor.assess_many(items)
            await db.close()
            db = Database(str(path))
            await db.connect()
            executor = RevisionWorkExecutor(client=client, model="fixture", store=db, derivation_id=root.id)
        result = await executor.assess_many(items)
        assert not result["w0"].supported
        assert executor.final_work_ids
        if failure:
            assert executor.reused == failure - 1
        assert await db.get_current_source_unit_revision(root.source_unit_id) is None
        rows = await db.db.execute_fetchall(
            "SELECT payload_json FROM source_derivation_work WHERE derivation_id=?", (root.id,)
        )
        works = [json.loads(r["payload_json"]) for r in rows]
        receipt = next(w for w in works if w["kind"] == "support_finalize")
        assert receipt["manifest"]["completion"] == "program"
        assert all(w["kind"] in {"support_assess", "support_finalize"} for w in works)
        assert len(receipt["manifest"]["dependencies"]) >= 3
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_artifact_bytes_bound_to_first_assessment_and_carried_refs():
    from types import SimpleNamespace
    from memforge.pipeline.revision_assessment import RevisionAssessmentContext
    from memforge.pipeline.projection_images import load_projection_images
    from memforge.pipeline.revision_work import SupportWorkItem
    from tests.test_revision_assessment import old_support, memory
    from tests.test_projected_lifecycle_integration import _projection_with_artifact

    base = _projection_with_artifact(
        run_id="image-base",
        payload=b"old",
        provider_revision="1",
        inference_eligible=True,
        body="Two reviewers approve US releases.",
    )
    body = "Two reviewers approve US releases.\n\n" + "\n\n".join(
        f"Routine note {i}: maintain ordinary settings." for i in range(200)
    )
    target = _projection_with_artifact(
        run_id="image-target",
        payload=b"new",
        provider_revision="2",
        inference_eligible=True,
        body=body,
        prior=base.source_unit_revisions[0],
        prior_observations={r.observation_id: r for r in base.observation_revisions},
    )
    reads = []

    def read(uri):
        reads.append(uri)
        return b"new"

    context = RevisionAssessmentContext(
        projection=target,
        base=base,
        access_context_hash="scope",
        image_loader=lambda ids: load_projection_images(
            projection=target, observation_ids=ids, document_store=SimpleNamespace(read_artifact=read)
        ),
    )

    class ArtifactClient(Client):
        image_stages = set()

        async def evaluate_revision_work(self, prompt, *, response_format, images=(), **kwargs):
            tag = "assessment"
            data = payload(prompt)
            rows = data["current"]["primary_candidates"] + data["current"]["required_only_candidates"]
            image_rows = [
                r for r in rows if any(isinstance(m, dict) and "image_source_observation_id" in m for m in r[2:])
            ]
            assert len(images) == len(image_rows)
            if image_rows:
                self.image_stages.add(tag)
                assert all(image.body == b"new" for image in images)
                assert {image.source_observation_id for image in images} == {
                    r[-1]["image_source_observation_id"] for r in image_rows
                }
            if image_rows:
                return SupportAssessmentResponse(
                    results=[
                        SupportAssessmentResult(
                            work_id=c["work_id"],
                            status="supported",
                            primary_ref=image_rows[0][0],
                            reason="Fixture judgment based on the current diagram.",
                        )
                        for c in data["claims"]
                    ]
                )
            return await super().evaluate_revision_work(
                prompt, response_format=response_format, images=images, **kwargs
            )

    client = ArtifactClient()
    result = await RevisionWorkExecutor(client=client, model="fixture").assess_many(
        [SupportWorkItem("w0", memory(), old_support(base), context)]
    )
    assert result["w0"].supported
    assert client.image_stages == {"assessment"}
    assert reads and all(uri.endswith("diagram-2.png") for uri in reads)
    parts = result["w0"].memory.resolved_evidence_selection.parts
    assert parts[0].anchor == next(f.anchor for f in context.full_fragments if f.kind.value == "artifact")


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["epoch", "memory"])
async def test_completed_assessments_cannot_commit_after_concurrent_state_change(tmp_path, drift):
    from memforge.memory.lifecycle_plan import (
        LifecyclePlan,
        ReconciliationScope,
        LifecycleGateState,
        CoverageProof,
        StaleGuard,
    )
    from memforge.storage.database import _lifecycle_memory_version
    from memforge.source_derivation import source_unit_derivation_context_from_payload

    text = "Two reviewers approve US releases.\n\n" + "\n\n".join(
        f"Routine note {i}: maintain settings." for i in range(150)
    )
    from tests.test_derivation_work import staged_fixture
    from tests.test_revision_assessment import old_support, memory
    from memforge.pipeline.revision_work import SupportWorkItem
    from memforge.pipeline.revision_assessment import RevisionAssessmentContext
    from memforge.pipeline.source_projection_adapters import project_source_item
    from memforge.models import ContentItem, RawContent, NormalizedContent
    from datetime import datetime, timezone

    base, _ = staged_fixture()
    item = ContentItem(
        item_id="doc",
        title="Stage fixture",
        source_url="https://example.test/doc",
        last_modified=datetime(2026, 9, 7, tzinfo=timezone.utc),
        content_type="text/markdown",
        version="2",
    )
    target = project_source_item(
        source_id="source-1",
        source_type="github_repo",
        run_id="run-2",
        item=item,
        raw=RawContent(item=item, body=text.encode(), content_type="text/markdown"),
        normalized=NormalizedContent(item=item, markdown_body=text),
        scope={},
        access_context={"visibility": "workspace"},
        prior_unit_revision=base.source_unit_revisions[0],
        prior_observation_revisions={r.observation_id: r for r in base.observation_revisions},
    )
    context = RevisionAssessmentContext(projection=target, base=base, access_context_hash="scope")
    items = [SupportWorkItem("w0", memory(), old_support(base), context)]
    db, root = await prepare_database(tmp_path / "drift.db", context.projection)
    try:
        await db.record_source_projection(context.base)
        await db.insert_memory(items[0].memory)
        epoch = await db.get_source_activity_epoch(root.source_id)
        rows = await db.db.execute_fetchall(
            "SELECT status,content_hash,updated_at FROM memories WHERE id = ?", (items[0].memory.id,)
        )
        version = _lifecycle_memory_version(rows[0])
        executor = RevisionWorkExecutor(client=Client(), model="fixture", store=db, derivation_id=root.id)
        await executor.assess_many(items)
        assert executor.final_work_ids and executor.stage_counts["support_assess"] > 1
        if drift == "epoch":
            await db.db.execute(
                "UPDATE sources SET activity_epoch = activity_epoch + 1 WHERE id = ?", (root.source_id,)
            )
        else:
            await db.db.execute(
                "UPDATE memories SET content_hash = ? WHERE id = ?", ("concurrent-change", items[0].memory.id)
            )
        await db.db.commit()
        delta = context.projection.deltas[0]
        plan = LifecyclePlan(
            id="drift-plan",
            scope=ReconciliationScope(
                id="scope",
                source_id=root.source_id,
                source_unit_id=root.source_unit_id,
                base_unit_revision_id=delta.previous_unit_revision_id,
                target_unit_revision_id=delta.current_unit_revision_id,
            ),
            gate_state=LifecycleGateState.GATED,
            coverage_proof=CoverageProof((), (), (), ()),
            stale_guard=StaleGuard((), {}, memory_versions={items[0].memory.id: version}),
            mutations=(),
        )
        with pytest.raises(ValueError, match="epoch|Memory stale guard"):
            await db.apply_source_projection_lifecycle(
                context.projection,
                plan,
                document=source_unit_derivation_context_from_payload(root.context_payload).document,
                derivation_id=root.id,
                derivation_context_identity_hash=root.context_identity_hash,
                required_derivation_work_ids=executor.final_work_ids,
                expected_source_activity_epoch=epoch,
            )
        assert (
            await db.get_current_source_unit_revision(root.source_unit_id)
        ).id == context.base.source_unit_revisions[0].id
        assert not await db.db.execute_fetchall("SELECT id FROM lifecycle_plans WHERE id = ?", (plan.id,))
    finally:
        await db.close()
