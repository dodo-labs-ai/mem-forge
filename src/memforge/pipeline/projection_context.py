"""Exact Claim Extraction authority and planned requests for one Source Unit revision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from memforge.pipeline.projection_fragments import ProjectionFragmentCatalog

from memforge.pipeline.evidence_fragments import (
    EvidenceFragment,
    canonical_nested_changed_raw_ranges,
    canonical_record_field_ranges,
    canonical_record_is_tombstoned,
    revision_changed_structural_ranges,
)
from memforge.source_artifacts import SOURCE_ARTIFACT_OBSERVATION_TYPE, source_artifact_inference_eligibility
from memforge.source_projection import (
    RevisionDelta,
    SourceObservationRevision,
    SourceProjection,
    SourceRelationType,
    SourceUnitRevision,
)


PROJECTION_AUTHORITY_SEGMENTATION_POLICY_VERSION = 6


@dataclass(frozen=True, slots=True)
class ExtractionAuthority:
    """The current structures Claim Extraction may cite as Primary in one Source Unit revision.

    Each authorized Observation maps to exact half-open ranges of its current
    revision, or to ``None`` when its whole revision is authorized. Reading
    context never widens this authority.
    """

    ranges_by_observation_id: Mapping[str, tuple[tuple[int, int], ...] | None]

    def authorizes(self, fragment: EvidenceFragment) -> bool:
        anchor = fragment.anchor
        if anchor.observation_id not in self.ranges_by_observation_id:
            return False
        ranges = self.ranges_by_observation_id[anchor.observation_id]
        if ranges is None:
            return True
        start, end = anchor.range_start, anchor.range_end
        return (
            start is not None
            and end is not None
            and any(left <= start and end <= right for left, right in ranges)
        )


@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    """One planned Claim Extraction request: whole ReadingGroups and the catalog they are read with."""

    id: str
    source_unit_id: str
    catalog: ProjectionFragmentCatalog
    prompt_sha256: str

    @property
    def primary_observation_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                fragment.anchor.observation_id
                for fragment in self.catalog.fragments
                if fragment.primary_eligible
            )
        )


class ProjectionEvidencePlanningFailureCode(str, Enum):
    INCREMENTAL_BASE_UNAVAILABLE = "INCREMENTAL_BASE_UNAVAILABLE"
    INCREMENTAL_AUTHORITY_UNMAPPABLE = "INCREMENTAL_AUTHORITY_UNMAPPABLE"
    CANONICAL_FIELD_MAPPING_INVALID = "CANONICAL_FIELD_MAPPING_INVALID"
    EVIDENCE_WORK_IDENTITY_INCOMPLETE = "EVIDENCE_WORK_IDENTITY_INCOMPLETE"
    REPRESENTATION_PROFILE_UNSUPPORTED = "REPRESENTATION_PROFILE_UNSUPPORTED"
    REPROCESS_AUTHORIZATION_MISSING = "REPROCESS_AUTHORIZATION_MISSING"


@dataclass(frozen=True, slots=True)
class ProjectionEvidencePlanningFailure:
    code: ProjectionEvidencePlanningFailureCode
    observation_id: str | None
    observation_revision_id: str | None
    representation_profile: str | None
    changed_structure_count: int = 0
    authorized_structure_count: int = 0


@dataclass(frozen=True, slots=True)
class CommittedSourceUnitSnapshot:
    unit_revision: SourceUnitRevision
    observation_revisions: tuple[SourceObservationRevision, ...]

    def __post_init__(self) -> None:
        revision_ids = {
            revision.id for revision in self.observation_revisions
        }
        if revision_ids != set(self.unit_revision.observation_revision_ids):
            raise ValueError(
                "committed Source Unit snapshot requires complete Observation membership"
            )

    @property
    def revisions_by_observation_id(
        self,
    ) -> Mapping[str, SourceObservationRevision]:
        return {
            revision.observation_id: revision
            for revision in self.observation_revisions
        }


def plan_projection_evidence_work(
    projection: SourceProjection,
    *,
    committed_base_snapshot: CommittedSourceUnitSnapshot | None = None,
    reprocess_all_current_observations: bool,
) -> ExtractionAuthority | ProjectionEvidencePlanningFailure:
    """Plan current-work Primary authority before any request is planned.

    Provider adapters declare representation and change facts. This compares
    the committed base and staged target through representation-owned
    structure; reading context and request packing cannot widen it.
    """

    delta = projection.deltas[0]
    revisions = {
        revision.observation_id: revision
        for revision in projection.observation_revisions
    }
    if reprocess_all_current_observations:
        if (
            delta.previous_unit_revision_id is not None
            and (
                committed_base_snapshot is None
                or committed_base_snapshot.unit_revision.id
                != delta.previous_unit_revision_id
                or committed_base_snapshot.unit_revision.source_unit_id
                != delta.source_unit_id
            )
        ):
            return _incremental_base_failure(delta, revisions)
        return whole_revision_extraction_authority(projection)

    changed_ids = [anchor.observation_id for anchor in delta.changed_anchors]
    changed_ids.extend(delta.added_observation_ids)
    authority: dict[str, tuple[tuple[int, int], ...] | None] = dict.fromkeys(changed_ids)
    if delta.previous_unit_revision_id is None:
        return _authority_or_failure(projection, authority)

    added_ids = set(delta.added_observation_ids)
    if (
        committed_base_snapshot is None
        or committed_base_snapshot.unit_revision.id
        != delta.previous_unit_revision_id
        or committed_base_snapshot.unit_revision.source_unit_id
        != delta.source_unit_id
    ):
        return _incremental_base_failure(delta, revisions)
    base_revisions = committed_base_snapshot.revisions_by_observation_id
    for anchor in delta.changed_anchors:
        if anchor.observation_id in added_ids:
            continue
        revision = revisions.get(anchor.observation_id)
        if revision is None:
            return ProjectionEvidencePlanningFailure(
                code=(
                    ProjectionEvidencePlanningFailureCode.INCREMENTAL_AUTHORITY_UNMAPPABLE
                ),
                observation_id=anchor.observation_id,
                observation_revision_id=anchor.observation_revision_id,
                representation_profile=None,
            )
        profile = revision.evidence_profile
        if profile is None:
            return ProjectionEvidencePlanningFailure(
                code=(
                    ProjectionEvidencePlanningFailureCode.REPRESENTATION_PROFILE_UNSUPPORTED
                ),
                observation_id=revision.observation_id,
                observation_revision_id=revision.id,
                representation_profile=None,
            )
        if profile.name == "canonical-record":
            base_revision = base_revisions.get(
                anchor.observation_id
            )
            if base_revision is None:
                return ProjectionEvidencePlanningFailure(
                    code=(
                        ProjectionEvidencePlanningFailureCode.INCREMENTAL_BASE_UNAVAILABLE
                    ),
                    observation_id=revision.observation_id,
                    observation_revision_id=revision.id,
                    representation_profile=profile.name,
                )
            try:
                if canonical_record_is_tombstoned(revision):
                    authority[anchor.observation_id] = ()
                    continue
                base_fields = {
                    item.descriptor.json_pointer: item
                    for item in canonical_record_field_ranges(base_revision)
                }
                target_fields = canonical_record_field_ranges(revision)
            except ValueError:
                return ProjectionEvidencePlanningFailure(
                    code=(
                        ProjectionEvidencePlanningFailureCode.CANONICAL_FIELD_MAPPING_INVALID
                    ),
                    observation_id=revision.observation_id,
                    observation_revision_id=revision.id,
                    representation_profile=profile.name,
                )
            changed_ranges = []
            for item in target_fields:
                base_field = base_fields.get(item.descriptor.json_pointer)
                if (
                    base_field is not None
                    and base_field.comparison_value == item.comparison_value
                ):
                    continue
                if base_field is not None and item.descriptor.nested_profile:
                    try:
                        changed_ranges.extend(
                            canonical_nested_changed_raw_ranges(base_field, item)
                        )
                    except ValueError:
                        return ProjectionEvidencePlanningFailure(
                            code=(
                                ProjectionEvidencePlanningFailureCode.CANONICAL_FIELD_MAPPING_INVALID
                            ),
                            observation_id=revision.observation_id,
                            observation_revision_id=revision.id,
                            representation_profile=profile.name,
                        )
                else:
                    changed_ranges.append((item.start, item.end))
            authority[anchor.observation_id] = tuple(changed_ranges)
            continue
        if profile.name not in {
            "markdown-structural",
            "plain-text",
        }:
            continue
        base_revision = base_revisions.get(anchor.observation_id)
        if base_revision is None:
            return ProjectionEvidencePlanningFailure(
                code=(
                    ProjectionEvidencePlanningFailureCode.INCREMENTAL_BASE_UNAVAILABLE
                ),
                observation_id=revision.observation_id,
                observation_revision_id=revision.id,
                representation_profile=profile.name,
            )
        try:
            authority[anchor.observation_id] = revision_changed_structural_ranges(
                base_revision,
                revision,
            )
        except ValueError:
            return ProjectionEvidencePlanningFailure(
                code=(
                    ProjectionEvidencePlanningFailureCode.INCREMENTAL_AUTHORITY_UNMAPPABLE
                ),
                observation_id=revision.observation_id,
                observation_revision_id=revision.id,
                representation_profile=profile.name,
            )

    return _authority_or_failure(projection, authority)


def whole_revision_extraction_authority(
    projection: SourceProjection,
) -> ExtractionAuthority | ProjectionEvidencePlanningFailure:
    """Authorize every current Observation, as a reprocess of the current revision does."""

    return _authority_or_failure(
        projection,
        dict.fromkeys(observation.id for observation in projection.observations),
    )


def _authority_or_failure(
    projection: SourceProjection,
    authority: Mapping[str, tuple[tuple[int, int], ...] | None],
) -> ExtractionAuthority | ProjectionEvidencePlanningFailure:
    """Keep authority on current Observations; a tombstoned canonical record authorizes nothing."""

    current = {
        revision.observation_id: revision
        for revision in projection.observation_revisions
    }
    authorized: dict[str, tuple[tuple[int, int], ...] | None] = {}
    for observation_id, ranges in authority.items():
        revision = current.get(observation_id)
        if revision is None or ranges == ():
            continue
        profile = revision.evidence_profile
        if profile is not None and profile.name == "canonical-record":
            try:
                if canonical_record_is_tombstoned(revision):
                    continue
            except ValueError:
                return ProjectionEvidencePlanningFailure(
                    code=ProjectionEvidencePlanningFailureCode.CANONICAL_FIELD_MAPPING_INVALID,
                    observation_id=revision.observation_id,
                    observation_revision_id=revision.id,
                    representation_profile=profile.name,
                )
        authorized[observation_id] = ranges
    return ExtractionAuthority(authorized)


def _incremental_base_failure(
    delta: RevisionDelta,
    revisions: Mapping[str, SourceObservationRevision],
) -> ProjectionEvidencePlanningFailure:
    first_changed = next(iter(delta.changed_anchors), None)
    target_revision = (
        revisions.get(first_changed.observation_id)
        if first_changed is not None
        else None
    )
    return ProjectionEvidencePlanningFailure(
        code=ProjectionEvidencePlanningFailureCode.INCREMENTAL_BASE_UNAVAILABLE,
        observation_id=(
            target_revision.observation_id
            if target_revision is not None
            else None
        ),
        observation_revision_id=(
            target_revision.id if target_revision is not None else None
        ),
        representation_profile=(
            target_revision.evidence_profile.name
            if target_revision is not None
            and target_revision.evidence_profile is not None
            else None
        ),
    )


def observation_is_inference_eligible(
    observation_type: str,
    metadata: dict,
) -> bool:
    if observation_type != SOURCE_ARTIFACT_OBSERVATION_TYPE:
        return True
    return source_artifact_inference_eligibility(metadata) is True


def preceding_observation_id(
    projection: SourceProjection,
    observation_id: str,
) -> str | None:
    """The Observation this one answers or follows by provider relation.

    A reply reads with the message it replies to; otherwise an Observation reads
    with its declared predecessor. Relations the provider did not declare are
    never inferred from order or similarity.
    """

    for relation_type, own_end in (
        (SourceRelationType.REPLIES_TO, "from_id"),
        (SourceRelationType.PRECEDES, "to_id"),
    ):
        for relation in projection.relations:
            if relation.relation_type is relation_type and getattr(relation, own_end) == observation_id:
                return relation.to_id if own_end == "from_id" else relation.from_id
    return None
