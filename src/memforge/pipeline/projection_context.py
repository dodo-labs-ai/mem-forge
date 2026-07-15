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
    context_observation_ids: tuple[str, ...]
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
    max_primary_chars: int = 60_000,
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

    related: dict[str, set[str]] = {item: set() for item in ordered_ids}
    for relation in projection.relations:
        if relation.from_id in related and relation.to_id in related:
            related[relation.from_id].add(relation.to_id)
            related[relation.to_id].add(relation.from_id)

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
        context_candidates: list[str] = []
        for observation_id in primary:
            position = ordered_ids.index(observation_id)
            if position > 0:
                context_candidates.append(ordered_ids[position - 1])
            if position + 1 < len(ordered_ids):
                context_candidates.append(ordered_ids[position + 1])
            context_candidates.extend(sorted(related[observation_id]))
        if ordered_ids:
            context_candidates.append(ordered_ids[0])
        primary_set = set(primary)
        context = tuple(
            dict.fromkeys(
                item for item in context_candidates if item not in primary_set and item in revisions
            )
        )
        primary_markdown = "\n\n".join(segment.markdown for segment in group)
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
                context_observation_ids=context,
                primary_markdown=primary_markdown,
                context_markdown=context_markdown,
            )
        )
    return tuple(batches)


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
