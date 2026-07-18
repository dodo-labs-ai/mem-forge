"""Provider-neutral extraction batches for changed Source Observations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from memforge.source_projection import SourceProjection


@dataclass(frozen=True, slots=True)
class ProjectionExtractionBatch:
    """Transient token-bounded work partition inside one Source Unit."""

    id: str
    source_unit_id: str
    primary_observation_ids: tuple[str, ...]
    primary_content_by_observation_id: tuple[tuple[str, str], ...]
    context_observation_ids: tuple[str, ...]
    context_observation_ids_by_primary: tuple[tuple[str, tuple[str, ...]], ...]
    primary_markdown: str
    context_markdown: str


@dataclass(frozen=True, slots=True)
class _PrimarySegment:
    observation_id: str
    start: int
    end: int
    markdown: str


def plan_projection_extraction_batches(
    projection: SourceProjection,
    *,
    max_primary_observations: int = 8,
    max_primary_chars: int = 30_000,
    max_context_chars: int = 20_000,
    primary_overlap_chars: int = 2_000,
) -> tuple[ProjectionExtractionBatch, ...]:
    """Build bounded batches using only generic deltas and relations.

    Changed/added observations are Primary. Directly related observations,
    immediate sequence neighbors, and the first observation in a unit are
    Context. Context is never promoted to extraction authority here.
    """

    if len(projection.source_units) != 1:
        raise ValueError("projection context planning requires exactly one Source Unit")
    unit = projection.source_units[0]
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
    primary_ids = tuple(item for item in ordered_ids if item in changed)
    if not primary_ids:
        return ()

    if max_primary_observations < 1 or max_primary_chars < 1 or max_context_chars < 0:
        raise ValueError("projection extraction budgets must be positive")
    if primary_overlap_chars < 0:
        raise ValueError("primary overlap cannot be negative")

    segments = [
        segment
        for observation_id in primary_ids
        for segment in _primary_segments(
            observation_id,
            observations[observation_id].observation_type,
            revisions[observation_id].content,
            max_chars=max_primary_chars,
            overlap_chars=primary_overlap_chars,
        )
    ]
    groups: list[list[_PrimarySegment]] = []
    current: list[_PrimarySegment] = []
    current_chars = 0
    for segment in segments:
        content_chars = len(segment.markdown)
        if current and (
            len(current) >= max_primary_observations
            or current_chars + content_chars > max_primary_chars
        ):
            groups.append(current)
            current = []
            current_chars = 0
        current.append(segment)
        current_chars += content_chars
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
        root_observation_id = ordered_ids[0] if ordered_ids else None
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
        segment_identity = "|".join(
            f"{segment.observation_id}:{segment.start}:{segment.end}" for segment in group
        )
        digest = hashlib.sha256(
            f"{projection.run_id}\x1f{unit.id}\x1f{index}\x1f{segment_identity}".encode()
        ).hexdigest()[:16]
        batches.append(
            ProjectionExtractionBatch(
                id=f"xbatch-{digest}",
                source_unit_id=unit.id,
                primary_observation_ids=primary,
                primary_content_by_observation_id=primary_content_by_observation_id,
                context_observation_ids=context,
                context_observation_ids_by_primary=context_by_primary,
                primary_markdown=primary_markdown,
                context_markdown=context_markdown,
            )
        )
    return tuple(batches)


def context_observation_ids_for(
    projection: SourceProjection,
    primary_observation_id: str,
) -> tuple[str, ...]:
    """Return deterministic claim context for one projected Observation."""

    revisions = {item.observation_id for item in projection.observation_revisions}
    ordered_ids = tuple(item.id for item in projection.observations if item.id in revisions)
    if primary_observation_id not in ordered_ids:
        return ()
    position = ordered_ids.index(primary_observation_id)
    candidates: list[str] = []
    if position > 0:
        candidates.append(ordered_ids[position - 1])
    if position + 1 < len(ordered_ids):
        candidates.append(ordered_ids[position + 1])
    for relation in projection.relations:
        if relation.from_id == primary_observation_id and relation.to_id in revisions:
            candidates.append(relation.to_id)
        elif relation.to_id == primary_observation_id and relation.from_id in revisions:
            candidates.append(relation.from_id)
    if ordered_ids and ordered_ids[0] != primary_observation_id:
        candidates.append(ordered_ids[0])
    return tuple(dict.fromkeys(candidates))


def _primary_segments(
    observation_id: str,
    observation_type: str,
    content: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> tuple[_PrimarySegment, ...]:
    """Slice one large Observation without changing its lifecycle identity."""

    plain_header = f"### Observation {observation_id} ({observation_type})\n"
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
