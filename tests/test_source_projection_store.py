from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from memforge.source_projection import (
    AnchorKind,
    DeltaAxis,
    FragmentMapping,
    ProjectionCoverage,
    ProjectionScopeTransition,
    ProjectionScopeTransitionStatus,
    RevisionDelta,
    SourceAnchor,
    SourceObservation,
    SourceObservationRevision,
    SourceProjection,
    SourceRelation,
    SourceRelationType,
    SourceUnit,
    SourceUnitRevision,
)
from memforge.models import DocumentMetadata, DocumentRecord, Memory
from memforge.storage.database import Database, MIGRATIONS


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "source-projection.db"))
    await database.connect()
    await database.upsert_source(
        id="src-1",
        type="confluence",
        name="Engineering",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    try:
        yield database
    finally:
        await database.close()


def _projection() -> SourceProjection:
    observation = SourceObservation(
        id="obs-page-1-body",
        source_id="src-1",
        source_unit_id="unit-page-1",
        observation_type="body",
        provider_key="page-1:body",
        locator={"page_id": "page-1"},
    )
    revision = SourceObservationRevision(
        id="obsrev-page-1-v2",
        observation_id=observation.id,
        semantic_hash="body-hash-v2",
        content="new body",
        observed_at="2026-07-15T00:00:00Z",
        metadata={"version": 2},
    )
    unit = SourceUnit(
        id="unit-page-1",
        source_id="src-1",
        unit_type="confluence_page",
        provider_key="page-1",
        locator={
            "url": "https://example.test/pages/page-1",
            "document_id": "confluence-page-1",
        },
    )
    unit_revision = SourceUnitRevision(
        id="unitrev-page-1-v2",
        source_unit_id=unit.id,
        semantic_hash="unit-hash-v2",
        location_hash="parent-b",
        observation_revision_ids=(revision.id,),
        observed_at="2026-07-15T00:00:00Z",
    )
    changed_anchor = SourceAnchor(
        kind=AnchorKind.STABLE_FRAGMENT,
        observation_id=observation.id,
        observation_revision_id=revision.id,
        fragment_id="section-results",
    )
    return SourceProjection(
        run_id="projection-run-1",
        source_id="src-1",
        source_type="confluence",
        scope={"spaces": ["ENG"]},
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        observations=(observation,),
        observation_revisions=(revision,),
        source_units=(unit,),
        source_unit_revisions=(unit_revision,),
        relations=(
            SourceRelation(
                relation_type=SourceRelationType.CONTAINED_BY,
                from_id=unit.id,
                to_id="unit-parent",
                provider_relation_id="page-1:parent",
                metadata={"position": 3},
            ),
        ),
        deltas=(
            RevisionDelta(
                source_unit_id=unit.id,
                previous_unit_revision_id="unitrev-page-1-v1",
                current_unit_revision_id=unit_revision.id,
                axes=frozenset({DeltaAxis.SEMANTIC, DeltaAxis.LOCATION}),
                coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
                changed_anchors=(changed_anchor,),
                fragment_mappings=(
                    FragmentMapping(
                        observation_id=observation.id,
                        previous_revision_id="obsrev-page-1-v1",
                        current_revision_id=revision.id,
                        previous_fragment_id="old-results",
                        current_fragment_id="section-results",
                    ),
                ),
            ),
        ),
        checkpoint={"cursor": "next-page"},
    )


def test_projection_schema_has_a_forward_migration() -> None:
    version, description, statements = next(item for item in MIGRATIONS if item[0] == 47)

    assert version == 47
    assert description == "Add durable Source Projection lineage"
    assert any("CREATE TABLE IF NOT EXISTS source_projection_runs" in item for item in statements)

    lineage_version, lineage_description, lineage_statements = next(
        item for item in MIGRATIONS if item[0] == 54
    )
    assert lineage_version == 54
    assert lineage_description == "Track Source Unit document lineage across moves"
    assert any("source_unit_document_lineage" in item for item in lineage_statements)


@pytest.mark.asyncio
async def test_source_projection_round_trips_as_one_atomic_record(db: Database) -> None:
    projection = _projection()

    await db.record_source_projection(projection)

    assert await db.get_source_projection(projection.run_id) == projection
    assert await db.get_current_source_unit_revision("unit-page-1") == projection.source_unit_revisions[0]


@pytest.mark.asyncio
async def test_identical_projection_retry_is_idempotent(db: Database) -> None:
    projection = _projection()

    await db.record_source_projection(projection)
    await db.record_source_projection(projection)

    assert await db.get_source_projection(projection.run_id) == projection


@pytest.mark.asyncio
async def test_projection_run_id_cannot_be_reused_for_different_payload(db: Database) -> None:
    projection = _projection()
    await db.record_source_projection(projection)

    with pytest.raises(ValueError, match="projection retry payload mismatch"):
        await db.record_source_projection(replace(projection, checkpoint={"cursor": "different"}))


@pytest.mark.asyncio
async def test_same_semantic_revision_can_be_reobserved_at_a_later_time(db: Database) -> None:
    initial = _projection()
    await db.record_source_projection(initial)
    later = replace(
        initial,
        run_id="projection-run-later",
        observation_revisions=(
            replace(initial.observation_revisions[0], observed_at="2026-07-16T00:00:00Z"),
        ),
        source_unit_revisions=(
            replace(initial.source_unit_revisions[0], observed_at="2026-07-16T00:00:00Z"),
        ),
        checkpoint={"cursor": "later"},
    )

    await db.record_source_projection(later)

    assert await db.get_source_projection(later.run_id) == later
    # The immutable semantic revision keeps its first-observed metadata while
    # the run payload records the later observation independently.
    assert await db.get_current_source_unit_revision("unit-page-1") == initial.source_unit_revisions[0]


@pytest.mark.asyncio
async def test_stable_unit_preserves_document_lineage_across_move(db: Database) -> None:
    initial = _projection()
    await db.record_source_projection(initial)
    moved_revision = replace(
        initial.source_unit_revisions[0],
        id="unitrev-page-1-moved",
        location_hash="parent-c",
    )
    moved = replace(
        initial,
        run_id="projection-run-moved",
        source_units=(
            replace(
                initial.source_units[0],
                locator={
                    "url": "https://example.test/new/page-1",
                    "document_id": "confluence-page-1-moved",
                },
            ),
        ),
        source_unit_revisions=(moved_revision,),
        deltas=(
            replace(
                initial.deltas[0],
                current_unit_revision_id=moved_revision.id,
                axes=frozenset({DeltaAxis.LOCATION}),
            ),
        ),
    )

    await db.record_source_projection(moved)

    assert await db.list_source_unit_document_ids("unit-page-1") == (
        "confluence-page-1-moved",
        "confluence-page-1",
    )
    current = await db.find_source_unit_by_document_id(
        "src-1",
        "confluence-page-1-moved",
    )
    historical = await db.find_source_unit_by_document_id(
        "src-1",
        "confluence-page-1",
    )
    assert current == moved.source_units[0]
    assert historical == moved.source_units[0]


@pytest.mark.asyncio
async def test_document_move_rebinds_legacy_support_without_cleaning_shared_artifacts(
    db: Database,
) -> None:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    shared_raw_uri = "/artifacts/raw/page-1.html"
    shared_normalized_uri = "/artifacts/normalized/page-1.md"
    for doc_id in ("old-path", "new-path"):
        await db.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                source="src-1",
                source_url=f"https://example.test/{doc_id}",
                title="Page",
                space_or_project="ENG",
                author=None,
                last_modified=now,
                labels=[],
                version="1",
                content_hash="same-content",
                token_count=10,
                raw_content_uri=shared_raw_uri,
                raw_content_type="text/html",
                normalized_content_uri=shared_normalized_uri,
                pdf_content_uri=None,
                last_synced=now,
            )
        )
    await db.upsert_metadata(
        DocumentMetadata(
            doc_id="new-path",
            summary="Page summary",
            tags=[],
            entities=[],
            doc_type="document",
            complexity="low",
            enriched_at=now,
        )
    )
    memory = Memory(
        id="mem-moved-page",
        memory_type="fact",
        content="The page records a durable fact.",
        content_hash="hash-moved-page",
    )
    await db.insert_memory(memory)
    await db.add_memory_source(
        memory.id,
        "old-path",
        "confluence",
        "durable fact",
        source_updated_at=now,
    )

    await db.rebind_projected_document_support("old-path", "new-path")
    await db.delete_projected_document("old-path")

    assert [source.doc_id for source in await db.get_memory_sources(memory.id)] == [
        "new-path"
    ]
    assert await db.get_document("old-path") is None
    assert await db.get_document("new-path") is not None
    cleanup_rows = await db.db.execute_fetchall(
        "SELECT artifact_uri FROM source_artifact_cleanup_tasks"
    )
    assert cleanup_rows == []


@pytest.mark.asyncio
async def test_tombstone_preserves_unit_history_and_clears_removed_observation_current_pointer(
    db: Database,
) -> None:
    initial = _projection()
    await db.record_source_projection(initial)
    unit = initial.source_units[0]
    prior_revision = initial.source_unit_revisions[0]
    tombstone_revision = SourceUnitRevision(
        id="unitrev-page-1-tombstone",
        source_unit_id=unit.id,
        semantic_hash="tombstone-hash",
        location_hash=prior_revision.location_hash,
        membership_hash="empty-membership-hash",
        observation_revision_ids=(),
        observed_at="2026-07-15T01:00:00Z",
    )
    tombstone = SourceProjection(
        run_id="projection-run-tombstone",
        source_id=initial.source_id,
        source_type=initial.source_type,
        scope=initial.scope,
        coverage=ProjectionCoverage.TOMBSTONED_DELTA,
        observations=(),
        observation_revisions=(),
        source_units=(unit,),
        source_unit_revisions=(tombstone_revision,),
        relations=(),
        deltas=(
            RevisionDelta(
                source_unit_id=unit.id,
                previous_unit_revision_id=prior_revision.id,
                current_unit_revision_id=tombstone_revision.id,
                axes=frozenset({DeltaAxis.SEMANTIC, DeltaAxis.MEMBERSHIP}),
                coverage=ProjectionCoverage.TOMBSTONED_DELTA,
                removed_observation_ids=tuple(
                    observation.id for observation in initial.observations
                ),
            ),
        ),
        checkpoint={"tombstoned": True},
    )

    await db.record_source_projection(tombstone)

    assert await db.get_current_source_unit_revision(unit.id) == tombstone_revision
    assert await db.get_current_source_observation_revisions(unit.id) == {}
    assert await db.get_source_projection(initial.run_id) == initial


@pytest.mark.asyncio
async def test_scope_transition_requires_complete_snapshot_before_apply(db: Database) -> None:
    transition = ProjectionScopeTransition(
        id="scope-transition-1",
        source_id="src-1",
        previous_scope={"spaces": ["OLD"]},
        target_scope={"spaces": ["NEW"]},
        created_at="2026-07-15T00:00:00+00:00",
    )

    created = await db.create_projection_scope_transition(transition)
    running = await db.start_projection_scope_transition(created.id, run_id="run-1")
    failed = await db.fail_projection_scope_transition(
        running.id,
        run_id="run-1",
        coverage=ProjectionCoverage.PARTIAL_PROJECTION,
        error="provider polling is partial",
    )

    assert failed.status is ProjectionScopeTransitionStatus.FAILED
    assert (await db.get_open_projection_scope_transition("src-1")) == failed

    await db.start_projection_scope_transition(failed.id, run_id="run-2")
    applied = await db.complete_projection_scope_transition(
        failed.id,
        run_id="run-2",
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
    )

    assert applied.status is ProjectionScopeTransitionStatus.APPLIED
    assert applied.coverage is ProjectionCoverage.COMPLETE_SNAPSHOT
    assert await db.get_open_projection_scope_transition("src-1") is None
    assert await db.list_projection_scope_transitions("src-1") == [applied]


@pytest.mark.asyncio
async def test_scope_transition_retry_identity_is_immutable(db: Database) -> None:
    transition = ProjectionScopeTransition(
        id="scope-transition-retry",
        source_id="src-1",
        previous_scope={"spaces": ["OLD"]},
        target_scope={"spaces": ["NEW"]},
    )
    await db.create_projection_scope_transition(transition)
    await db.create_projection_scope_transition(transition)

    with pytest.raises(ValueError, match="retry identity mismatch"):
        await db.create_projection_scope_transition(
            replace(transition, target_scope={"spaces": ["OTHER"]})
        )
