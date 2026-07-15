from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.source_projection import (
    AnchorKind,
    DeltaAxis,
    FragmentMapping,
    ImpactResult,
    ProjectionCoverage,
    ProjectionEnvelope,
    ProjectionRequest,
    ProjectionRunMode,
    RevisionDelta,
    SourceAnchor,
    SourceObservation,
    SourceObservationRevision,
    SourceProjection,
    SourceProjectionAdapter,
    SourceRelation,
    SourceRelationType,
    SourceUnit,
    SourceUnitRevision,
    resolve_anchor_impact,
)


def _anchor(
    *,
    observation_id: str = "obs-page-1-body",
    revision_id: str = "obsrev-2",
    fragment_id: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> SourceAnchor:
    if fragment_id is not None:
        return SourceAnchor(
            kind=AnchorKind.STABLE_FRAGMENT,
            observation_id=observation_id,
            observation_revision_id=revision_id,
            fragment_id=fragment_id,
        )
    if start is not None:
        return SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id=observation_id,
            observation_revision_id=revision_id,
            range_start=start,
            range_end=end,
        )
    return SourceAnchor(
        kind=AnchorKind.WHOLE_OBSERVATION,
        observation_id=observation_id,
        observation_revision_id=revision_id,
    )


def test_anchor_shape_is_controlled() -> None:
    with pytest.raises(ValueError, match="fragment_id"):
        SourceAnchor(
            kind=AnchorKind.STABLE_FRAGMENT,
            observation_id="obs-1",
            observation_revision_id="rev-1",
        )

    with pytest.raises(ValueError, match="range"):
        SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id="obs-1",
            observation_revision_id="rev-1",
            range_start=10,
            range_end=5,
        )


def test_partial_projection_cannot_claim_absence() -> None:
    with pytest.raises(ValueError, match="removed_observation_ids"):
        RevisionDelta(
            source_unit_id="unit-1",
            previous_unit_revision_id="unitrev-1",
            current_unit_revision_id="unitrev-2",
            axes=frozenset({DeltaAxis.MEMBERSHIP}),
            coverage=ProjectionCoverage.PARTIAL_PROJECTION,
            removed_observation_ids=("obs-removed",),
        )


def test_whole_observation_anchor_is_never_disjoint_from_same_observation_change() -> None:
    delta = RevisionDelta(
        source_unit_id="unit-1",
        previous_unit_revision_id="unitrev-1",
        current_unit_revision_id="unitrev-2",
        axes=frozenset({DeltaAxis.SEMANTIC}),
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        changed_anchors=(_anchor(fragment_id="section-new"),),
    )

    assert resolve_anchor_impact(_anchor(), delta) is ImpactResult.UNKNOWN


def test_fragment_mapping_can_prove_affected_or_disjoint() -> None:
    delta = RevisionDelta(
        source_unit_id="unit-1",
        previous_unit_revision_id="unitrev-1",
        current_unit_revision_id="unitrev-2",
        axes=frozenset({DeltaAxis.SEMANTIC}),
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        changed_anchors=(_anchor(fragment_id="section-b"),),
        fragment_mappings=(
            FragmentMapping(
                observation_id="obs-page-1-body",
                previous_revision_id="obsrev-1",
                current_revision_id="obsrev-2",
                previous_fragment_id="section-a-old",
                current_fragment_id="section-a",
            ),
        ),
    )

    assert (
        resolve_anchor_impact(
            _anchor(revision_id="obsrev-1", fragment_id="section-a-old"),
            delta,
        )
        is ImpactResult.DISJOINT
    )
    assert resolve_anchor_impact(_anchor(fragment_id="section-b"), delta) is ImpactResult.AFFECTED


def test_range_overlap_is_generic() -> None:
    delta = RevisionDelta(
        source_unit_id="unit-1",
        previous_unit_revision_id="unitrev-1",
        current_unit_revision_id="unitrev-2",
        axes=frozenset({DeltaAxis.SEMANTIC}),
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        changed_anchors=(_anchor(start=20, end=30),),
    )

    assert resolve_anchor_impact(_anchor(start=25, end=35), delta) is ImpactResult.AFFECTED
    assert resolve_anchor_impact(_anchor(start=0, end=10), delta) is ImpactResult.DISJOINT


def test_location_only_delta_does_not_affect_semantic_evidence() -> None:
    delta = RevisionDelta(
        source_unit_id="unit-page-1",
        previous_unit_revision_id="unitrev-1",
        current_unit_revision_id="unitrev-2",
        axes=frozenset({DeltaAxis.LOCATION}),
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
    )

    assert resolve_anchor_impact(_anchor(), delta) is ImpactResult.DISJOINT
    assert delta.requires_extraction is False


def test_projection_rejects_cross_unit_observation_revision_lineage() -> None:
    observation = SourceObservation(
        id="obs-1",
        source_id="src-1",
        source_unit_id="unit-1",
        observation_type="body",
        provider_key="page-1:body",
    )
    revision = SourceObservationRevision(
        id="obsrev-1",
        observation_id=observation.id,
        semantic_hash="body-hash",
        content="body",
    )
    units = (
        SourceUnit("unit-1", "src-1", "page", "page-1"),
        SourceUnit("unit-2", "src-1", "page", "page-2"),
    )
    with pytest.raises(ValueError, match="another unit"):
        SourceProjection(
            run_id="run-invalid-lineage",
            source_id="src-1",
            source_type="confluence",
            scope={},
            coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
            observations=(observation,),
            observation_revisions=(revision,),
            source_units=units,
            source_unit_revisions=(
                SourceUnitRevision(
                    id="unitrev-2",
                    source_unit_id="unit-2",
                    semantic_hash="unit-hash",
                    observation_revision_ids=(revision.id,),
                ),
            ),
            relations=(),
            deltas=(),
            checkpoint={},
        )


def test_projection_rejects_delta_anchor_outside_projected_revision() -> None:
    observation = SourceObservation(
        id="obs-1",
        source_id="src-1",
        source_unit_id="unit-1",
        observation_type="body",
        provider_key="page-1:body",
    )
    revision = SourceObservationRevision(
        id="obsrev-1",
        observation_id=observation.id,
        semantic_hash="body-hash",
        content="body",
    )
    unit = SourceUnit("unit-1", "src-1", "page", "page-1")
    unit_revision = SourceUnitRevision(
        id="unitrev-1",
        source_unit_id=unit.id,
        semantic_hash="unit-hash",
        observation_revision_ids=(revision.id,),
    )
    with pytest.raises(ValueError, match="anchor"):
        SourceProjection(
            run_id="run-invalid-anchor",
            source_id="src-1",
            source_type="confluence",
            scope={},
            coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
            observations=(observation,),
            observation_revisions=(revision,),
            source_units=(unit,),
            source_unit_revisions=(unit_revision,),
            relations=(),
            deltas=(
                RevisionDelta(
                    source_unit_id=unit.id,
                    previous_unit_revision_id=None,
                    current_unit_revision_id=unit_revision.id,
                    axes=frozenset({DeltaAxis.SEMANTIC}),
                    coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
                    changed_anchors=(
                        SourceAnchor(
                            kind=AnchorKind.WHOLE_OBSERVATION,
                            observation_id=observation.id,
                            observation_revision_id="obsrev-not-projected",
                        ),
                    ),
                ),
            ),
            checkpoint={},
        )


class _Adapter:
    async def project(self, envelope: ProjectionEnvelope) -> SourceProjection:
        request = envelope.request
        observation = SourceObservation(
            id="obs-page-1-body",
            source_id=request.source_id,
            source_unit_id="unit-page-1",
            observation_type="body",
            provider_key="page-1:body",
        )
        revision = SourceObservationRevision(
            id="obsrev-2",
            observation_id=observation.id,
            semantic_hash="sha256:new",
            content="new body",
        )
        unit = SourceUnit(
            id="unit-page-1",
            source_id=request.source_id,
            unit_type="confluence_page",
            provider_key="page-1",
        )
        unit_revision = SourceUnitRevision(
            id="unitrev-2",
            source_unit_id=unit.id,
            semantic_hash="sha256:new",
            observation_revision_ids=(revision.id,),
        )
        return SourceProjection(
            run_id="projection-run-2",
            source_id=request.source_id,
            source_type=request.source_type,
            scope=request.scope,
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
                ),
            ),
            deltas=(),
            checkpoint={"cursor": "next"},
        )


@pytest.mark.asyncio
async def test_adapter_contract_is_one_provider_neutral_projection() -> None:
    adapter: SourceProjectionAdapter = _Adapter()
    projection = await adapter.project(
        ProjectionEnvelope(
            request=ProjectionRequest(
                run_id="projection-run-2",
                source_id="src-confluence",
                source_type="confluence",
                scope={"space_keys": ["ENG"]},
                run_mode=ProjectionRunMode.FULL_SNAPSHOT,
            ),
            item=(
                item := ContentItem(
                    item_id="page-1",
                    title="Page 1",
                    source_url="https://example.test/pages/1",
                    space_or_project="ENG",
                    version="2",
                    last_modified=datetime.now(timezone.utc),
                )
            ),
            raw=RawContent(item=item, body=b"body", content_type="text/plain"),
            normalized=NormalizedContent(item=item, markdown_body="body"),
        )
    )

    assert projection.source_units[0].id == "unit-page-1"
    assert projection.relations[0].relation_type is SourceRelationType.CONTAINED_BY
    assert projection.checkpoint == {"cursor": "next"}
