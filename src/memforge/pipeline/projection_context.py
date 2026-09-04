"""Provider-neutral extraction batches for changed Source Observations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from memforge.pipeline.extraction_contract import (
    PROJECTION_EXTRACTION_CONTRACT_VERSION,
    projection_extraction_contract,
)
from memforge.pipeline.evidence_fragments import (
    StructuralUnit,
    StructuralUnitTooLargeError,
    canonical_nested_changed_raw_ranges,
    canonical_record_field_ranges,
    canonical_record_is_tombstoned,
    plan_revision_structural_units,
    revision_changed_structural_ranges,
    revision_structural_ranges,
)
from memforge.source_artifacts import (
    MAX_SOURCE_ARTIFACT_INFERENCE_BYTES_PER_BATCH,
    source_artifact_inference_eligibility,
)
from memforge.source_projection import (
    RevisionDelta,
    SourceObservationRevision,
    SourceProjection,
    SourceUnitRevision,
)


LEGACY_PROJECTION_AUTHORITY_SEGMENTATION_POLICY_VERSION = 2
PROJECTION_AUTHORITY_SEGMENTATION_POLICY_VERSION = 5


@dataclass(frozen=True, slots=True)
class ProjectionExtractionBatch:
    """Transient token-bounded work partition inside one Source Unit."""

    id: str
    source_unit_id: str
    primary_image_bytes: int
    primary_observation_ids: tuple[str, ...]
    primary_content_by_observation_id: tuple[tuple[str, str], ...]
    context_observation_ids: tuple[str, ...]
    context_observation_ids_by_primary: tuple[tuple[str, tuple[str, ...]], ...]
    primary_markdown: str
    context_markdown: str
    authority_policy_version: int = PROJECTION_AUTHORITY_SEGMENTATION_POLICY_VERSION
    # Exact segment coordinates in immutable Observation revisions. Kept
    # transient so EvidenceCatalog never creates a block across overlap seams.
    primary_authority_spans: tuple[tuple[str, int, str], ...] = ()
    candidate_context_observation_ids: tuple[str, ...] | None = None
    candidate_context_image_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _PrimarySegment:
    observation_id: str
    start: int
    end: int
    markdown: str


class ProjectionEvidencePlanningFailureCode(str, Enum):
    INCREMENTAL_BASE_UNAVAILABLE = "INCREMENTAL_BASE_UNAVAILABLE"
    INCREMENTAL_AUTHORITY_UNMAPPABLE = "INCREMENTAL_AUTHORITY_UNMAPPABLE"
    CANONICAL_FIELD_MAPPING_INVALID = "CANONICAL_FIELD_MAPPING_INVALID"
    EVIDENCE_WORK_IDENTITY_INCOMPLETE = "EVIDENCE_WORK_IDENTITY_INCOMPLETE"
    REPRESENTATION_PROFILE_UNSUPPORTED = "REPRESENTATION_PROFILE_UNSUPPORTED"
    REPROCESS_AUTHORIZATION_MISSING = "REPROCESS_AUTHORIZATION_MISSING"
    STRUCTURAL_UNIT_TOO_LARGE = "STRUCTURAL_UNIT_TOO_LARGE"


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
    extraction_contract_version: str = PROJECTION_EXTRACTION_CONTRACT_VERSION,
) -> tuple[ProjectionExtractionBatch, ...] | ProjectionEvidencePlanningFailure:
    """Plan current-work authority before presentation batching.

    Provider adapters declare representation and change facts. This active
    fragment-catalog seam compares the committed base and staged target through
    representation-owned structure; Context and batch packing cannot widen it.
    """

    contract = projection_extraction_contract(extraction_contract_version)
    delta = projection.deltas[0]
    revisions = {
        revision.observation_id: revision
        for revision in projection.observation_revisions
    }
    selected_primary_ids = (
        tuple(observation.id for observation in projection.observations)
        if reprocess_all_current_observations
        else None
    )
    if not contract.uses_fragment_catalog or reprocess_all_current_observations:
        if not contract.uses_fragment_catalog:
            return plan_projection_extraction_batches(
                projection,
                primary_observation_ids=selected_primary_ids,
                extraction_contract_version=extraction_contract_version,
            )
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
        return _plan_projection_extraction_batches_or_failure(
            projection,
            primary_observation_ids=selected_primary_ids,
            primary_authority_ranges_by_observation_id=None,
            extraction_contract_version=extraction_contract_version,
        )

    if delta.previous_unit_revision_id is None:
        tombstoned_ranges: dict[str, tuple[tuple[int, int], ...]] = {}
        for observation_id in delta.added_observation_ids:
            revision = revisions.get(observation_id)
            if revision is None:
                return ProjectionEvidencePlanningFailure(
                    code=(
                        ProjectionEvidencePlanningFailureCode.INCREMENTAL_AUTHORITY_UNMAPPABLE
                    ),
                    observation_id=observation_id,
                    observation_revision_id=None,
                    representation_profile=None,
                )
            profile = revision.evidence_profile
            if profile is None or profile.name != "canonical-record":
                continue
            try:
                tombstoned = canonical_record_is_tombstoned(revision)
            except ValueError:
                return ProjectionEvidencePlanningFailure(
                    code=(
                        ProjectionEvidencePlanningFailureCode.CANONICAL_FIELD_MAPPING_INVALID
                    ),
                    observation_id=observation_id,
                    observation_revision_id=revision.id,
                    representation_profile=profile.name,
                )
            if tombstoned:
                tombstoned_ranges[observation_id] = ()
        return _plan_projection_extraction_batches_or_failure(
            projection,
            primary_authority_ranges_by_observation_id=(
                tombstoned_ranges if tombstoned_ranges else None
            ),
            extraction_contract_version=extraction_contract_version,
        )

    added_ids = set(delta.added_observation_ids)
    exact_authority_ranges: dict[str, tuple[tuple[int, int], ...]] = {}
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
                    exact_authority_ranges[anchor.observation_id] = ()
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
            exact_authority_ranges[anchor.observation_id] = tuple(changed_ranges)
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
            mapped_ranges = revision_changed_structural_ranges(
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
        exact_authority_ranges[anchor.observation_id] = mapped_ranges

    return _plan_projection_extraction_batches_or_failure(
        projection,
        primary_authority_ranges_by_observation_id=exact_authority_ranges,
        extraction_contract_version=extraction_contract_version,
    )


def _plan_projection_extraction_batches_or_failure(
    projection: SourceProjection,
    *,
    primary_observation_ids: tuple[str, ...] | None = None,
    primary_authority_ranges_by_observation_id: Mapping[
        str, tuple[tuple[int, int], ...]
    ]
    | None,
    extraction_contract_version: str,
) -> tuple[ProjectionExtractionBatch, ...] | ProjectionEvidencePlanningFailure:
    """Keep deterministic presentation limits inside the typed planner contract."""

    try:
        return plan_projection_extraction_batches(
            projection,
            primary_observation_ids=primary_observation_ids,
            primary_authority_ranges_by_observation_id=(
                primary_authority_ranges_by_observation_id
            ),
            extraction_contract_version=extraction_contract_version,
        )
    except StructuralUnitTooLargeError as exc:
        revision = next(
            (
                candidate
                for candidate in projection.observation_revisions
                if candidate.id == exc.revision_id
            ),
            None,
        )
        return ProjectionEvidencePlanningFailure(
            code=ProjectionEvidencePlanningFailureCode.STRUCTURAL_UNIT_TOO_LARGE,
            observation_id=(revision.observation_id if revision is not None else None),
            observation_revision_id=exc.revision_id,
            representation_profile=(
                revision.evidence_profile.name
                if revision is not None and revision.evidence_profile is not None
                else None
            ),
            changed_structure_count=1,
        )


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


def plan_projection_extraction_batches(
    projection: SourceProjection,
    *,
    primary_observation_ids: tuple[str, ...] | None = None,
    primary_authority_ranges_by_observation_id: Mapping[
        str, tuple[tuple[int, int], ...]
    ]
    | None = None,
    max_primary_observations: int = 8,
    max_primary_chars: int = 30_000,
    max_context_chars: int = 20_000,
    primary_overlap_chars: int = 2_000,
    max_primary_binary_bytes: int = MAX_SOURCE_ARTIFACT_INFERENCE_BYTES_PER_BATCH,
    extraction_contract_version: str = PROJECTION_EXTRACTION_CONTRACT_VERSION,
) -> tuple[ProjectionExtractionBatch, ...]:
    """Build bounded batches using only generic deltas and relations.

    Changed/added observations are Primary-eligible by default. A bounded
    operator replay may explicitly select current observations without changing
    Source Projection truth. Directly related observations, immediate sequence
    neighbors, and the first observation in a unit are bounded Context. Exact
    candidate Context may be selected as Required, but relations never make it
    Primary-eligible. Compiler-backed v9 planning segments only range-addressable
    text profiles; canonical records and binary Artifacts retain whole-Observation
    authority until compilation. Legacy projection extraction keeps its bounded
    character segmentation because it presents batch Markdown directly.
    """

    if len(projection.source_units) != 1:
        raise ValueError("projection context planning requires exactly one Source Unit")
    if len(projection.source_unit_revisions) != 1:
        raise ValueError(
            "projection context planning requires exactly one Source Unit revision"
        )
    unit = projection.source_units[0]
    target_unit_revision_id = projection.source_unit_revisions[0].id
    revisions = {item.observation_id: item for item in projection.observation_revisions}
    observations = {item.id: item for item in projection.observations}
    ordered_ids = tuple(item.id for item in projection.observations if item.id in revisions)
    changed = {
        anchor.observation_id
        for delta in projection.deltas
        for anchor in delta.changed_anchors
    }
    changed.update(
        observation_id
        for delta in projection.deltas
        for observation_id in delta.added_observation_ids
    )
    eligible_ordered_ids = tuple(
        item
        for item in ordered_ids
        if observation_is_inference_eligible(
            observations[item].observation_type,
            revisions[item].metadata,
        )
    )
    if primary_observation_ids is None:
        selected_primary_ids = changed
    else:
        requested_primary_ids = set(primary_observation_ids)
        unknown_primary_ids = requested_primary_ids - set(ordered_ids)
        if unknown_primary_ids:
            raise ValueError(
                "projection extraction Primary observations must belong to the current projection"
            )
        selected_primary_ids = requested_primary_ids
    primary_ids = tuple(
        item for item in eligible_ordered_ids if item in selected_primary_ids
    )
    if not primary_ids:
        return ()

    if (
        max_primary_observations < 1
        or max_primary_chars < 1
        or max_context_chars < 0
        or max_primary_binary_bytes < 1
    ):
        raise ValueError("projection extraction budgets must be positive")
    if primary_overlap_chars < 0:
        raise ValueError("primary overlap cannot be negative")

    compiler_backed = projection_extraction_contract(
        extraction_contract_version
    ).uses_fragment_catalog
    if primary_authority_ranges_by_observation_id is not None and not compiler_backed:
        raise ValueError(
            "exact Primary authority ranges require a fragment-catalog extraction contract"
        )
    if primary_authority_ranges_by_observation_id is not None:
        unknown_authority_ids = set(primary_authority_ranges_by_observation_id) - set(
            primary_ids
        )
        if unknown_authority_ids:
            raise ValueError(
                "Primary authority ranges must belong to selected current observations"
            )
    authority_policy_version = (
        PROJECTION_AUTHORITY_SEGMENTATION_POLICY_VERSION
        if compiler_backed
        else LEGACY_PROJECTION_AUTHORITY_SEGMENTATION_POLICY_VERSION
    )

    segments = [
        segment
        for observation_id in primary_ids
        for segment in _primary_segments(
            observation_id,
            observations[observation_id].observation_type,
            revisions[observation_id],
            preserve_whole_authority=compiler_backed,
            authorized_ranges=(
                primary_authority_ranges_by_observation_id.get(observation_id)
                if primary_authority_ranges_by_observation_id is not None
                else None
            ),
            max_chars=max_primary_chars,
            overlap_chars=primary_overlap_chars,
        )
    ]
    groups: list[list[_PrimarySegment]] = []
    current: list[_PrimarySegment] = []
    current_chars = 0
    current_binary_bytes = 0
    current_observation_ids: set[str] = set()
    for segment in segments:
        content_chars = len(segment.markdown)
        new_binary_bytes = (
            _observation_binary_size(revisions[segment.observation_id].metadata)
            if segment.observation_id not in current_observation_ids
            else 0
        )
        if current and (
            (
                segment.observation_id not in current_observation_ids
                and len(current_observation_ids) >= max_primary_observations
            )
            or current_chars + content_chars > max_primary_chars
            or current_binary_bytes + new_binary_bytes > max_primary_binary_bytes
        ):
            groups.append(current)
            current = []
            current_chars = 0
            current_binary_bytes = 0
            current_observation_ids = set()
            new_binary_bytes = _observation_binary_size(
                revisions[segment.observation_id].metadata
            )
        if new_binary_bytes > max_primary_binary_bytes:
            raise ValueError("one inference-eligible Artifact exceeds the batch byte budget")
        current.append(segment)
        current_chars += content_chars
        current_binary_bytes += new_binary_bytes
        current_observation_ids.add(segment.observation_id)
    if current:
        groups.append(current)

    batches = []
    for index, group in enumerate(groups):
        primary = tuple(dict.fromkeys(segment.observation_id for segment in group))
        primary_set = set(primary)
        context_candidates_by_primary = tuple(
            (
                observation_id,
                tuple(
                    item
                    for item in context_observation_ids_for(projection, observation_id)
                    if item not in primary_set
                ),
            )
            for observation_id in primary
        )
        root_observation_id = (
            eligible_ordered_ids[0] if eligible_ordered_ids else None
        )
        context_candidates = [
            item
            for _, candidates in context_candidates_by_primary
            for item in candidates
            if item != root_observation_id
        ]
        if root_observation_id is not None and root_observation_id not in primary_set:
            context_candidates.append(root_observation_id)
        context = tuple(
            dict.fromkeys(
                item for item in context_candidates if item in revisions
            )
        )
        context_set = set(context)
        context_by_primary = tuple(
            (
                observation_id,
                tuple(item for item in candidates if item in context_set),
            )
            for observation_id, candidates in context_candidates_by_primary
        )
        primary_markdown = "\n\n".join(segment.markdown for segment in group)
        primary_content_by_observation_id = tuple(
            (
                observation_id,
                "\n".join(
                    revisions[observation_id].content[segment.start : segment.end]
                    for segment in group
                    if segment.observation_id == observation_id
                ),
            )
            for observation_id in primary
        )
        context_markdown = _observation_markdown(context, observations, revisions)[:max_context_chars]
        presented_context: list[str] = []
        presented_context_chars = 0
        for observation_id in context:
            rendered = _observation_markdown(
                (observation_id,),
                observations,
                revisions,
            )
            separator_chars = 2 if presented_context else 0
            if (
                presented_context_chars + separator_chars + len(rendered)
                > max_context_chars
            ):
                break
            presented_context.append(observation_id)
            presented_context_chars += separator_chars + len(rendered)
        segment_identity = "|".join(
            f"{segment.observation_id}:{segment.start}:{segment.end}" for segment in group
        )
        digest = hashlib.sha256(
            (
                f"{extraction_contract_version}\x1f"
                f"authority-policy:{authority_policy_version}\x1f"
                f"{target_unit_revision_id}\x1f{unit.id}\x1f"
                f"{index}\x1f{segment_identity}"
            ).encode()
        ).hexdigest()[:16]
        primary_binary_bytes = sum(
            _observation_binary_size(revisions[observation_id].metadata)
            for observation_id in primary
        )
        remaining_context_binary_bytes = max(
            0,
            max_primary_binary_bytes - primary_binary_bytes,
        )
        candidate_context: list[str] = []
        candidate_context_binary_bytes = 0
        for observation_id in presented_context:
            if observations[observation_id].observation_type != "binary_artifact":
                candidate_context.append(observation_id)
                continue
            binary_bytes = _observation_binary_size(revisions[observation_id].metadata)
            if (
                candidate_context_binary_bytes + binary_bytes
                <= remaining_context_binary_bytes
            ):
                candidate_context.append(observation_id)
                candidate_context_binary_bytes += binary_bytes
        candidate_context_ids = tuple(candidate_context)
        batches.append(
            ProjectionExtractionBatch(
                id=f"xbatch-{digest}",
                source_unit_id=unit.id,
                primary_image_bytes=sum(
                    _observation_image_size(revisions[observation_id].metadata)
                    for observation_id in primary
                ),
                primary_observation_ids=primary,
                primary_content_by_observation_id=primary_content_by_observation_id,
                context_observation_ids=context,
                context_observation_ids_by_primary=context_by_primary,
                primary_markdown=primary_markdown,
                context_markdown=context_markdown,
                authority_policy_version=authority_policy_version,
                primary_authority_spans=tuple(
                    (
                        segment.observation_id,
                        segment.start,
                        revisions[segment.observation_id].content[
                            segment.start : segment.end
                        ],
                    )
                    for segment in group
                ),
                candidate_context_observation_ids=candidate_context_ids,
                candidate_context_image_bytes=sum(
                    _observation_image_size(revisions[observation_id].metadata)
                    for observation_id in candidate_context_ids
                ),
            )
        )
    return tuple(batches)


def observation_is_inference_eligible(
    observation_type: str,
    metadata: dict,
) -> bool:
    if observation_type != "binary_artifact":
        return True
    return source_artifact_inference_eligibility(metadata) is True


def _observation_binary_size(metadata: dict) -> int:
    raw = metadata.get("source_artifact")
    if not isinstance(raw, dict):
        return 0
    try:
        size_bytes = int(raw.get("size_bytes") or 0)
    except (TypeError, ValueError):
        return 0
    return max(size_bytes, 0)


def _observation_image_size(metadata: dict) -> int:
    raw = metadata.get("source_artifact")
    if not isinstance(raw, dict):
        return 0
    media_type = str(raw.get("media_type") or "").lower()
    if not media_type.startswith("image/"):
        return 0
    return _observation_binary_size(metadata)


def context_observation_ids_for(
    projection: SourceProjection,
    primary_observation_id: str,
) -> tuple[str, ...]:
    """Return deterministic claim context for one projected Observation."""

    revisions_by_observation = {
        item.observation_id: item for item in projection.observation_revisions
    }
    observations_by_id = {item.id: item for item in projection.observations}
    eligible_ids = {
        observation_id
        for observation_id, revision in revisions_by_observation.items()
        if observation_id in observations_by_id
        and observation_is_inference_eligible(
            observations_by_id[observation_id].observation_type,
            revision.metadata,
        )
    }
    ordered_ids = tuple(
        item.id for item in projection.observations if item.id in eligible_ids
    )
    if primary_observation_id not in ordered_ids:
        return ()
    position = ordered_ids.index(primary_observation_id)
    candidates: list[str] = []
    if position > 0:
        candidates.append(ordered_ids[position - 1])
    if position + 1 < len(ordered_ids):
        candidates.append(ordered_ids[position + 1])
    for relation in projection.relations:
        if relation.from_id == primary_observation_id and relation.to_id in eligible_ids:
            candidates.append(relation.to_id)
        elif relation.to_id == primary_observation_id and relation.from_id in eligible_ids:
            candidates.append(relation.from_id)
    if ordered_ids and ordered_ids[0] != primary_observation_id:
        candidates.append(ordered_ids[0])
    return tuple(dict.fromkeys(candidates))


def _primary_segments(
    observation_id: str,
    observation_type: str,
    revision: SourceObservationRevision,
    *,
    preserve_whole_authority: bool,
    authorized_ranges: tuple[tuple[int, int], ...] | None,
    max_chars: int,
    overlap_chars: int,
) -> tuple[_PrimarySegment, ...]:
    """Plan exact authority without violating the Revision's representation."""

    content = revision.content
    evidence_profile = revision.evidence_profile
    plain_header = f"### Observation {observation_id} ({observation_type})\n"
    if authorized_ranges is not None:
        if evidence_profile is None:
            raise ValueError(
                "exact text authority ranges require a range-addressable representation"
            )
        for start, end in authorized_ranges:
            if start < 0 or end <= start or end > len(content):
                raise ValueError("Primary authority contains an invalid target Revision range")
        if evidence_profile.name == "canonical-record":
            selected_units = tuple(
                StructuralUnit(start=start, end=end)
                for start, end in authorized_ranges
            )
        elif evidence_profile.name in {"markdown-structural", "plain-text"}:
            selected_units = tuple(
                unit
                for unit in revision_structural_ranges(revision)
                if any(
                    start < unit.end and unit.start < end
                    for start, end in authorized_ranges
                )
            )
        else:
            raise ValueError(
                "exact authority ranges require a range-addressable representation"
            )
        segments = []
        for unit in selected_units:
            header = (
                f"### Observation {observation_id} ({observation_type}) "
                f"[characters {unit.start}:{unit.end}]\n"
            )
            content_budget = max_chars - len(header)
            if unit.end - unit.start > content_budget:
                raise StructuralUnitTooLargeError(
                    revision_id=revision.id,
                    start=unit.start,
                    end=unit.end,
                    budget=max(content_budget, 0),
                )
            segments.append(
                _PrimarySegment(
                    observation_id=observation_id,
                    start=unit.start,
                    end=unit.end,
                    markdown=header + content[unit.start : unit.end],
                )
            )
        return tuple(segments)
    if (
        preserve_whole_authority
        and evidence_profile is not None
        and evidence_profile.requires_whole_observation_authority
    ):
        return (_PrimarySegment(observation_id, 0, len(content), plain_header + content),)
    if len(plain_header) + len(content) <= max_chars:
        return (_PrimarySegment(observation_id, 0, len(content), plain_header + content),)

    max_digits = len(str(len(content)))
    ranged_header = (
        f"### Observation {observation_id} ({observation_type}) "
        f"[characters {'9' * max_digits}:{'9' * max_digits}]\n"
    )
    content_budget = max_chars - len(ranged_header)
    if content_budget < 1:
        raise ValueError("primary character budget is too small for the Observation header")
    if preserve_whole_authority:
        return tuple(
            _PrimarySegment(
                observation_id=observation_id,
                start=unit.start,
                end=unit.end,
                markdown=(
                    f"### Observation {observation_id} ({observation_type}) "
                    f"[characters {unit.start}:{unit.end}]\n"
                    f"{content[unit.start:unit.end]}"
                ),
            )
            for unit in plan_revision_structural_units(
                revision,
                max_content_chars=content_budget,
            )
        )
    overlap = min(overlap_chars, content_budget // 4)
    step = content_budget - overlap
    segments = []
    start = 0
    while start < len(content):
        end = min(len(content), start + content_budget)
        header = (
            f"### Observation {observation_id} ({observation_type}) "
            f"[characters {start}:{end}]\n"
        )
        segments.append(
            _PrimarySegment(
                observation_id=observation_id,
                start=start,
                end=end,
                markdown=header + content[start:end],
            )
        )
        if end == len(content):
            break
        start += step
    return tuple(segments)


def _observation_markdown(observation_ids, observations, revisions) -> str:
    blocks = []
    for observation_id in observation_ids:
        observation = observations[observation_id]
        revision = revisions[observation_id]
        blocks.append(
            f"### Observation {observation_id} ({observation.observation_type})\n{revision.content}"
        )
    return "\n\n".join(blocks)
