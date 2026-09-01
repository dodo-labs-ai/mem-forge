"""Provider-neutral extraction batches for changed Source Observations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from memforge.pipeline.extraction_contract import (
    PROJECTION_EXTRACTION_CONTRACT_VERSION,
    projection_extraction_contract,
)
from memforge.source_artifacts import (
    MAX_SOURCE_ARTIFACT_INFERENCE_BYTES_PER_BATCH,
    source_artifact_inference_eligibility,
)
from memforge.source_projection import EvidenceRepresentationProfile, SourceProjection


LEGACY_PROJECTION_AUTHORITY_SEGMENTATION_POLICY_VERSION = 2
PROJECTION_AUTHORITY_SEGMENTATION_POLICY_VERSION = 3


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


def plan_projection_extraction_batches(
    projection: SourceProjection,
    *,
    primary_observation_ids: tuple[str, ...] | None = None,
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
            revisions[observation_id].content,
            evidence_profile=revisions[observation_id].evidence_profile,
            preserve_whole_authority=compiler_backed,
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
            len(current) >= max_primary_observations
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
    content: str,
    *,
    evidence_profile: EvidenceRepresentationProfile | None,
    preserve_whole_authority: bool,
    max_chars: int,
    overlap_chars: int,
) -> tuple[_PrimarySegment, ...]:
    """Plan exact authority without violating the Revision's representation."""

    plain_header = f"### Observation {observation_id} ({observation_type})\n"
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
