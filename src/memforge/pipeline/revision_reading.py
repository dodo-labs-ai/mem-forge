"""Representation-owned reading context for one immutable Revision.

Reading groups widen what a model may read, never what it may cite as Primary
Evidence.  Every group therefore refers to the existing atomic Fragment
anchors and keeps caller-selected authority separate from added context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from memforge.pipeline.evidence_fragments import (
    CanonicalFieldRange,
    EvidenceFragment,
    _OffsetHTMLParser,
    _line_starts,
    _markdown_parser,
    _token_range,
    _walk_html_nodes,
    canonical_record_field_ranges,
)
from memforge.source_projection import SourceAnchor, SourceObservationRevision
from memforge.source_representation import representation_contract_for_profile


@dataclass(frozen=True, slots=True)
class ReadingGroup:
    """One directional, representation-proven reading relationship."""

    kind: str
    range_start: int
    range_end: int
    trigger_anchors: tuple[SourceAnchor, ...]
    context_anchors: tuple[SourceAnchor, ...]
    owner: str | None = None
    # An outermost list: its member Fragments are read as one unit.
    is_list: bool = False


@dataclass(frozen=True, slots=True)
class ReadingExpansion:
    """Source-ordered Fragments with explicit authority/context provenance."""

    fragments: tuple[EvidenceFragment, ...]
    authority_anchors: tuple[SourceAnchor, ...]
    context_anchors: tuple[SourceAnchor, ...]


@dataclass(frozen=True, slots=True)
class RevisionReadingIndex:
    """Pure, operation-local reading context for one Observation Revision."""

    observation_revision_id: str
    fragments: tuple[EvidenceFragment, ...]
    groups: tuple[ReadingGroup, ...]

    @property
    def lists(self) -> tuple[ReadingGroup, ...]:
        """The outermost lists, in source order; no two of them share a Fragment."""

        return tuple(group for group in self.groups if group.is_list)

    def expand(self, selected: Sequence[EvidenceFragment]) -> ReadingExpansion:
        """Add representation-owned context without altering Fragment authority.

        Context is derived only from groups triggered by the caller's original
        selection.  Added context cannot recursively trigger unrelated groups.
        """

        if any(
            fragment.anchor.observation_revision_id != self.observation_revision_id
            for fragment in selected
        ):
            raise ValueError("selected Fragment belongs to another Observation Revision")

        selected_by_anchor = {fragment.anchor: fragment for fragment in selected}
        authority = _ordered_unique(tuple(selected_by_anchor))
        authority_set = set(authority)
        context = _ordered_unique(
            tuple(
                anchor
                for group in self.groups
                if authority_set.intersection(group.trigger_anchors)
                for anchor in group.context_anchors
                if anchor not in authority_set
            )
        )
        indexed_by_anchor = {fragment.anchor: fragment for fragment in self.fragments}
        combined = dict(indexed_by_anchor)
        combined.update(selected_by_anchor)
        expanded = tuple(
            sorted(
                (
                    combined[anchor]
                    for anchor in (*authority, *context)
                    if anchor in combined
                ),
                key=_fragment_order_key,
            )
        )
        present = {fragment.anchor for fragment in expanded}
        return ReadingExpansion(
            fragments=expanded,
            authority_anchors=tuple(anchor for anchor in authority if anchor in present),
            context_anchors=tuple(anchor for anchor in context if anchor in present),
        )


@dataclass(frozen=True, slots=True)
class _Heading:
    level: int
    start: int
    end: int
    kind: str


@dataclass(frozen=True, slots=True)
class _ListContainer:
    start: int
    end: int
    kind: str


@dataclass(frozen=True, slots=True)
class _Coordinates:
    """Map decoded nested text coordinates to immutable raw Revision offsets."""

    base: int = 0
    boundaries: tuple[int, ...] | None = None

    def range(self, start: int, end: int) -> tuple[int, int]:
        if self.boundaries is None:
            return self.base + start, self.base + end
        return self.boundaries[start], self.boundaries[end]


def build_revision_reading_index(
    revision: SourceObservationRevision,
    fragments: tuple[EvidenceFragment, ...],
) -> RevisionReadingIndex:
    """Build deterministic reading groups from the registered representation.

    The declared Evidence profile is the only format selector.  Source type,
    filename, MIME type, and semantic inference do not participate.
    """

    if any(fragment.anchor.observation_revision_id != revision.id for fragment in fragments):
        raise ValueError("Fragment index contains another Observation Revision")
    ordered_fragments = tuple(sorted(fragments, key=_fragment_order_key))
    contract = representation_contract_for_profile(revision.evidence_profile)
    if contract is None or not ordered_fragments:
        return RevisionReadingIndex(revision.id, ordered_fragments, ())

    profile_name = contract.profile.name
    groups: list[ReadingGroup] = []
    if profile_name == "markdown-structural":
        groups.extend(
            _text_groups(
                revision.content,
                ordered_fragments,
                coordinates=_Coordinates(),
            )
        )
    elif profile_name == "canonical-record":
        fields = canonical_record_field_ranges(revision)
        groups.extend(_canonical_context_groups(fields, ordered_fragments))
        for field in fields:
            if (
                field.descriptor.nested_profile == "markdown-structural"
                and isinstance(field.value, str)
                and field.string_boundaries is not None
            ):
                groups.extend(
                    _text_groups(
                        field.value,
                        ordered_fragments,
                        coordinates=_Coordinates(boundaries=field.string_boundaries),
                        kind_prefix="canonical-",
                        owner=field.descriptor.json_pointer,
                        raw_bounds=(field.start, field.end),
                    )
                )

    return RevisionReadingIndex(
        observation_revision_id=revision.id,
        fragments=ordered_fragments,
        groups=tuple(
            sorted(
                _deduplicate_groups(groups),
                key=lambda group: (
                    group.range_start,
                    group.range_end,
                    group.kind,
                    group.owner or "",
                ),
            )
        ),
    )


def _text_groups(
    text: str,
    fragments: tuple[EvidenceFragment, ...],
    *,
    coordinates: _Coordinates,
    kind_prefix: str = "",
    owner: str | None = None,
    raw_bounds: tuple[int, int] | None = None,
) -> tuple[ReadingGroup, ...]:
    try:
        tokens = _markdown_parser().parse(text)
    except Exception:
        return ()
    line_starts = _line_starts(text)
    headings: list[_Heading] = []
    lists: list[_ListContainer] = []
    html_regions: list[tuple[int, int]] = []

    for token in tokens:
        if token.map is None:
            continue
        start, end = _token_range(text, line_starts, token.map[0], token.map[1])
        if start >= end:
            continue
        raw_start, raw_end = coordinates.range(start, end)
        if token.type == "heading_open":
            headings.append(
                _Heading(
                    level=int(token.tag.removeprefix("h")),
                    start=raw_start,
                    end=raw_end,
                    kind=f"{kind_prefix}markdown-section",
                )
            )
        elif token.type in {"bullet_list_open", "ordered_list_open"}:
            lists.append(
                _ListContainer(
                    start=raw_start,
                    end=raw_end,
                    kind=f"{kind_prefix}markdown-list",
                )
            )
        elif token.type == "html_block":
            html_regions.append((start, end))

    for start, end in html_regions:
        parser = _OffsetHTMLParser(text[start:end])
        parser.close_checked()
        if parser.failure is not None:
            continue
        for node in _walk_html_nodes(parser.roots):
            if node.end is None:
                continue
            raw_start, raw_end = coordinates.range(start + node.start, start + node.end)
            if node.tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                headings.append(
                    _Heading(
                        level=int(node.tag[1:]),
                        start=raw_start,
                        end=raw_end,
                        kind=f"{kind_prefix}html-section",
                    )
                )
            elif node.tag in {"ul", "ol"}:
                lists.append(
                    _ListContainer(
                        start=raw_start,
                        end=raw_end,
                        kind=f"{kind_prefix}html-list",
                    )
                )

    region_start, region_end = raw_bounds or coordinates.range(0, len(text))
    headings.sort(key=lambda item: (item.start, item.end, item.level, item.kind))
    outer_lists = _outermost_lists(lists)
    groups = [
        *_heading_groups(
            headings,
            fragments,
            region_end=region_end,
            owner=owner,
        ),
        *_list_groups(
            outer_lists,
            headings,
            fragments,
            region_start=region_start,
            owner=owner,
        ),
    ]
    return tuple(groups)


def _heading_groups(
    headings: list[_Heading],
    fragments: tuple[EvidenceFragment, ...],
    *,
    region_end: int,
    owner: str | None,
) -> tuple[ReadingGroup, ...]:
    groups: list[ReadingGroup] = []
    for index, heading in enumerate(headings):
        scope_end = next(
            (
                candidate.start
                for candidate in headings[index + 1 :]
                if candidate.level <= heading.level
            ),
            region_end,
        )
        triggers = _anchors_in_range(fragments, heading.start, scope_end)
        heading_anchors = _anchors_in_range(
            fragments,
            heading.start,
            heading.end,
            exact=True,
        )
        if not triggers or not heading_anchors:
            continue
        next_heading_start = next(
            (candidate.start for candidate in headings[index + 1 :] if candidate.start < scope_end),
            scope_end,
        )
        intro = _leading_intro_anchors(
            fragments,
            start=heading.end,
            end=next_heading_start,
        )
        groups.append(
            ReadingGroup(
                kind=heading.kind,
                range_start=heading.start,
                range_end=scope_end,
                trigger_anchors=triggers,
                context_anchors=_ordered_unique((*heading_anchors, *intro)),
                owner=owner,
            )
        )
    return tuple(groups)


def _list_groups(
    lists: tuple[_ListContainer, ...],
    headings: list[_Heading],
    fragments: tuple[EvidenceFragment, ...],
    *,
    region_start: int,
    owner: str | None,
) -> tuple[ReadingGroup, ...]:
    groups: list[ReadingGroup] = []
    for container in lists:
        members = _anchors_in_range(fragments, container.start, container.end)
        if not members:
            continue
        lead_in = _list_lead_in(
            fragments,
            headings,
            list_start=container.start,
            region_start=region_start,
        )
        groups.append(
            ReadingGroup(
                kind=container.kind,
                range_start=container.start,
                range_end=container.end,
                trigger_anchors=members,
                context_anchors=_ordered_unique((*lead_in, *members)),
                owner=owner,
                is_list=True,
            )
        )
    return tuple(groups)


def _canonical_context_groups(
    fields: tuple[CanonicalFieldRange, ...],
    fragments: tuple[EvidenceFragment, ...],
) -> tuple[ReadingGroup, ...]:
    contextual = tuple(field for field in fields if field.descriptor.contextual)
    groups: list[ReadingGroup] = []
    for owner in fields:
        if owner.descriptor.contextual:
            continue
        triggers = _anchors_in_range(fragments, owner.start, owner.end)
        if not triggers:
            continue
        parent = owner.descriptor.json_pointer.rsplit("/", 1)[0]
        contexts = _ordered_unique(
            tuple(
                anchor
                for field in contextual
                if field.descriptor.json_pointer.rsplit("/", 1)[0] in {"", parent}
                for anchor in _anchors_in_range(fragments, field.start, field.end)
            )
        )
        if not contexts:
            continue
        groups.append(
            ReadingGroup(
                kind="canonical-field-context",
                range_start=owner.start,
                range_end=owner.end,
                trigger_anchors=triggers,
                context_anchors=contexts,
                owner=owner.descriptor.json_pointer,
            )
        )
    return tuple(groups)


def _leading_intro_anchors(
    fragments: tuple[EvidenceFragment, ...],
    *,
    start: int,
    end: int,
) -> tuple[SourceAnchor, ...]:
    candidates = [
        fragment
        for fragment in fragments
        if _fragment_inside(fragment, start, end)
    ]
    if candidates and _is_intro_fragment(candidates[0]):
        # A local intro is the first complete paragraph under the heading.  It
        # is deliberately not every direct paragraph: one broad root heading
        # must not turn an otherwise packable document into one reading group.
        return (candidates[0].anchor,)
    return ()


def _list_lead_in(
    fragments: tuple[EvidenceFragment, ...],
    headings: list[_Heading],
    *,
    list_start: int,
    region_start: int,
) -> tuple[SourceAnchor, ...]:
    previous = [
        fragment
        for fragment in fragments
        if (end := fragment.anchor.range_end) is not None
        and region_start <= end <= list_start
    ]
    if not previous:
        return ()
    candidate = previous[-1]
    if not _is_intro_fragment(candidate):
        return ()
    candidate_end = candidate.anchor.range_end
    assert candidate_end is not None
    if any(candidate_end <= heading.start < list_start for heading in headings):
        return ()
    if any(
        other.anchor.range_start is not None
        and candidate_end < other.anchor.range_start < list_start
        for other in fragments
    ):
        return ()
    return (candidate.anchor,)


def _outermost_lists(lists: Iterable[_ListContainer]) -> tuple[_ListContainer, ...]:
    ordered = sorted(lists, key=lambda item: (item.start, -item.end, item.kind))
    selected: list[_ListContainer] = []
    for candidate in ordered:
        if any(
            parent.start <= candidate.start
            and candidate.end <= parent.end
            for parent in selected
        ):
            continue
        selected.append(candidate)
    return tuple(selected)


def _anchors_in_range(
    fragments: tuple[EvidenceFragment, ...],
    start: int,
    end: int,
    *,
    exact: bool = False,
) -> tuple[SourceAnchor, ...]:
    return tuple(
        fragment.anchor
        for fragment in fragments
        if (
            fragment.anchor.range_start == start
            and fragment.anchor.range_end == end
            if exact
            else _fragment_inside(fragment, start, end)
        )
    )


def _fragment_inside(fragment: EvidenceFragment, start: int, end: int) -> bool:
    anchor = fragment.anchor
    return (
        anchor.range_start is not None
        and anchor.range_end is not None
        and start <= anchor.range_start
        and anchor.range_end <= end
    )


def _is_intro_fragment(fragment: EvidenceFragment) -> bool:
    kind = fragment.fragment_type
    return (
        kind.endswith("markdown-paragraph")
        or kind.endswith("markdown-inline-html")
        or kind.endswith("html-p")
    )


def _deduplicate_groups(groups: Iterable[ReadingGroup]) -> tuple[ReadingGroup, ...]:
    unique: dict[
        tuple[str, int, int, tuple[SourceAnchor, ...], tuple[SourceAnchor, ...], str | None],
        ReadingGroup,
    ] = {}
    for group in groups:
        key = (
            group.kind,
            group.range_start,
            group.range_end,
            group.trigger_anchors,
            group.context_anchors,
            group.owner,
        )
        unique[key] = group
    return tuple(unique.values())


def _ordered_unique(anchors: tuple[SourceAnchor, ...]) -> tuple[SourceAnchor, ...]:
    return tuple(sorted(set(anchors), key=_anchor_order_key))


def _fragment_order_key(fragment: EvidenceFragment) -> tuple[int, int, str]:
    start, end, suffix = _anchor_order_key(fragment.anchor)
    return start, end, f"{suffix}:{fragment.reference}"


def _anchor_order_key(anchor: SourceAnchor) -> tuple[int, int, str]:
    return (
        anchor.range_start if anchor.range_start is not None else -1,
        anchor.range_end if anchor.range_end is not None else -1,
        anchor.fragment_id or "",
    )
