"""Exact prior Evidence correspondence, whole-Support routing and reading order for one revision.

Everything here is program-only and recomputed for every base/target pair from
the persisted Evidence digests, the target membership, its coverage and its
reading structure. No correspondence status is stored, no semantic similarity
is consulted, and no cost comparison decides what a Support reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole
from memforge.models import Memory
from memforge.pipeline.evidence_fragments import EvidenceFragment
from memforge.pipeline.projection_fragments import ProjectionFragmentCatalog
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.source_projection import ProjectionCoverage, SourceAnchor


@dataclass(frozen=True)
class SupportWorkItem:
    """One fixed claim's Support in one Evidence Unit, against its validation baseline."""

    id: str
    memory: Memory
    support: tuple[ActiveSupportEvidence, ...]
    context: RevisionAssessmentContext

    def claim_payload(self):
        memory = self.memory
        return {
            "work_id": self.id,
            "claim": memory.content,
            "memory_type": memory.memory_type,
            "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
            "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
        }


class EvidenceCorrespondence(str, Enum):
    """How one prior Evidence part corresponds to the target revision."""

    EXACT_UNCHANGED = "exact_unchanged"
    MODIFIED = "modified"
    REMOVED = "removed"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PartCorrespondence:
    evidence: ActiveSupportEvidence
    status: EvidenceCorrespondence
    # The current Fragments this part is read with. EXACT_UNCHANGED: the one exact
    # Fragment; AMBIGUOUS: every exact candidate; MODIFIED inside an Observation
    # revision that is still current: the Fragments overlapping the old anchor;
    # otherwise empty.
    current: tuple[EvidenceFragment, ...] = ()


class SupportRoute(str, Enum):
    """What happens to one whole Support in this revision."""

    REBIND_SUPPORT = "rebind_support"
    SUPPORT_ASSESSMENT = "support_assessment"
    UNRESOLVED_PARTIAL_COVERAGE = "unresolved_partial_coverage"


@dataclass(frozen=True)
class SupportPlan:
    item: SupportWorkItem
    parts: tuple[PartCorrespondence, ...]
    route: SupportRoute

    @property
    def matched_refs(self) -> tuple[str, ...]:
        """Current catalog refs of the exactly unchanged parts, in part order."""
        return tuple(
            correspondence.current[0].reference
            for correspondence in self.parts
            if correspondence.status is EvidenceCorrespondence.EXACT_UNCHANGED
        )

    @property
    def unknown_observation_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            correspondence.evidence.anchor.observation_id
            for correspondence in self.parts
            if correspondence.status is EvidenceCorrespondence.UNKNOWN
        ))


@dataclass(frozen=True)
class ReadingPart:
    """One unit of the reading order: a current ReadingGroup, or one removed old Fragment's text."""

    # Selectable current Fragments, in document order; a whole list is one part.
    fragments: tuple[EvidenceFragment, ...] = ()
    # Removed old text with its historical ``ref``; never selectable.
    removed: Mapping[str, Any] | None = None

    @property
    def label(self) -> str:
        """A stable name for diagnostics."""
        if self.removed is not None:
            entry = self.removed
            return f"removed:{entry['observation_id']}:{entry['revision_id']}:{entry['ref']}"
        first, last = self.fragments[0].anchor, self.fragments[-1].anchor
        if first.range_start is None or last.range_end is None:
            return first.observation_id
        return f"{first.observation_id}:{first.range_start}-{last.range_end}"


@dataclass(frozen=True)
class SupportReadingOrder:
    """One fixed order over the complete current revision for every assessed Support of a cohort."""

    parts: tuple[ReadingPart, ...]
    # Work id -> how many leading parts form that Support's first part.
    first_part_end: Mapping[str, int]

    @property
    def removed(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(part.removed for part in self.parts if part.removed is not None)

    @property
    def has_current_content(self) -> bool:
        return any(part.fragments for part in self.parts)


@dataclass(frozen=True)
class SupportRevisionPlan:
    # The complete current Support catalog; every current ref here belongs to it.
    catalog: ProjectionFragmentCatalog
    supports: tuple[SupportPlan, ...]
    # Present when any Support routes to Support Assessment.
    reading: SupportReadingOrder | None


def plan_support_revision(
    context: RevisionAssessmentContext, items: Sequence[SupportWorkItem]
) -> SupportRevisionPlan:
    """Correspond and route every fixed Support that shares one base/target pair."""
    catalog = context.catalog(context.full_fragments)
    by_observation: dict[str, list[EvidenceFragment]] = {}
    for fragment in catalog.fragments:
        by_observation.setdefault(fragment.anchor.observation_id, []).append(fragment)
    returned = frozenset(observation.id for observation in context.projection.observations)
    correspondences = [
        tuple(_correspond(part, context, returned, by_observation) for part in item.support) for item in items
    ]
    # Changed content only matters for a Support that is exactly unchanged against a usable baseline.
    changed_content = (
        context.base is not None
        and any(_all_exact(parts) for parts in correspondences)
        and _has_changed_content(context)
    )
    supports = tuple(
        SupportPlan(item, parts, _route(parts, context, changed_content))
        for item, parts in zip(items, correspondences, strict=True)
    )
    assessed = tuple(support for support in supports if support.route is SupportRoute.SUPPORT_ASSESSMENT)
    reading = _reading_order(context, catalog, assessed) if assessed else None
    return SupportRevisionPlan(catalog, supports, reading)


def _correspond(
    part: ActiveSupportEvidence,
    context: RevisionAssessmentContext,
    returned: frozenset[str],
    by_observation: dict[str, list[EvidenceFragment]],
) -> PartCorrespondence:
    observation_id = part.anchor.observation_id
    coverage = context.projection.coverage
    # Another Source Unit's Observation is never a member here, so it is never rebound.
    if observation_id not in context.members:
        status = EvidenceCorrespondence.REMOVED if coverage.proves_absence else EvidenceCorrespondence.UNKNOWN
        return PartCorrespondence(part, status)
    if observation_id in context.tombstoned:
        return PartCorrespondence(part, EvidenceCorrespondence.REMOVED)
    # A carried Observation the provider did not return proves neither presence nor absence.
    if coverage is ProjectionCoverage.PARTIAL_PROJECTION and observation_id not in returned:
        return PartCorrespondence(part, EvidenceCorrespondence.UNKNOWN)
    candidates = tuple(fragment for fragment in by_observation.get(observation_id, ()) if _is_exact(part, fragment))
    if len(candidates) == 1:
        return PartCorrespondence(part, EvidenceCorrespondence.EXACT_UNCHANGED, candidates)
    if candidates:
        return PartCorrespondence(part, EvidenceCorrespondence.AMBIGUOUS, candidates)
    # The Observation is still an authoritative member, but the exact text is gone. When
    # its revision is unchanged (a legacy part without digests, or a recompiled split),
    # the old anchor still locates the current text it is read with.
    overlapping = tuple(
        fragment for fragment in by_observation.get(observation_id, ()) if _overlaps(part.anchor, fragment.anchor)
    )
    return PartCorrespondence(part, EvidenceCorrespondence.MODIFIED, overlapping)


def _overlaps(old: SourceAnchor, current: SourceAnchor) -> bool:
    """Whether a current Fragment lies on an old anchor of the same Observation revision."""
    if current.observation_revision_id != old.observation_revision_id:
        return False
    if old.fragment_id is not None and current.fragment_id == old.fragment_id:
        return True
    ranges = (old.range_start, old.range_end, current.range_start, current.range_end)
    return None not in ranges and current.range_start < old.range_end and old.range_start < current.range_end


def _is_exact(part: ActiveSupportEvidence, fragment: EvidenceFragment) -> bool:
    """Compare every persisted digest without normalization; legacy parts without one never match."""
    if part.role is EvidenceRole.PRIMARY and not fragment.primary_eligible:
        return False
    recorded = tuple(
        (old, new)
        for old, new in (
            (part.raw_content_sha256, fragment.raw_content_sha256),
            (part.presentation_sha256, fragment.presentation_sha256),
        )
        if old is not None
    )
    return bool(recorded) and all(old == new for old, new in recorded)


def _all_exact(parts: tuple[PartCorrespondence, ...]) -> bool:
    return all(correspondence.status is EvidenceCorrespondence.EXACT_UNCHANGED for correspondence in parts)


def _has_changed_content(context: RevisionAssessmentContext) -> bool:
    changed, removed = context.delta()
    # Removed old text is changed content: it may have qualified the claim.
    return bool(changed) or bool(removed)


def _route(
    parts: tuple[PartCorrespondence, ...],
    context: RevisionAssessmentContext,
    changed_content: bool,
) -> SupportRoute:
    # An UNKNOWN part never reaches the model, so it decides the whole Support.
    if any(correspondence.status is EvidenceCorrespondence.UNKNOWN for correspondence in parts):
        return SupportRoute.UNRESOLVED_PARTIAL_COVERAGE
    # Without a usable baseline nothing is known to be unchanged: read the whole revision.
    if context.base is None or not _all_exact(parts) or changed_content:
        return SupportRoute.SUPPORT_ASSESSMENT
    return SupportRoute.REBIND_SUPPORT


def _reading_order(
    context: RevisionAssessmentContext,
    catalog: ProjectionFragmentCatalog,
    assessed: tuple[SupportPlan, ...],
) -> SupportReadingOrder:
    """Changed content, removed old text and own prior Evidence first; then the rest, in document order."""
    groups = _current_reading_groups(context, catalog)
    if context.base is None:
        # Nothing is known to be unchanged, so every Support's first part is the whole revision.
        parts = tuple(ReadingPart(fragments=group) for group in groups)
        return SupportReadingOrder(parts, {support.item.id: len(parts) for support in assessed})

    changed_fragments, removed_entries = context.delta()
    changed_anchors = {fragment.anchor for fragment in changed_fragments}
    own_anchors = {
        fragment.anchor for support in assessed for correspondence in support.parts for fragment in correspondence.current
    }
    changed = [index for index, group in enumerate(groups) if any(f.anchor in changed_anchors for f in group)]
    first = set(changed)
    own = [index for index, group in enumerate(groups) if index not in first and any(f.anchor in own_anchors for f in group)]
    first.update(own)
    rest = [index for index in range(len(groups)) if index not in first]
    removed = tuple(
        ReadingPart(removed={"ref": f"h{index:06d}", **entry}) for index, entry in enumerate(removed_entries)
    )
    parts = (
        *(ReadingPart(fragments=groups[index]) for index in changed),
        *removed,
        *(ReadingPart(fragments=groups[index]) for index in (*own, *rest)),
    )
    changed_end = len(changed) + len(removed)
    position = {
        fragment.anchor: index for index, part in enumerate(parts) for fragment in part.fragments
    }

    def first_part_end(support: SupportPlan) -> int:
        own_ends = (
            position[fragment.anchor] + 1 for correspondence in support.parts for fragment in correspondence.current
        )
        # Prior Evidence travels with the first part, so every Support reads at least one part before concluding.
        return max((1, changed_end, *own_ends))

    return SupportReadingOrder(parts, {support.item.id: first_part_end(support) for support in assessed})


def _current_reading_groups(
    context: RevisionAssessmentContext, catalog: ProjectionFragmentCatalog
) -> tuple[tuple[EvidenceFragment, ...], ...]:
    """Partition the current catalog in document order: one outermost list, or one Fragment, per group."""
    list_of: dict[SourceAnchor, tuple[str, int]] = {}
    for revision in context.current.values():
        for index, group in enumerate(context.reading_index(revision).lists):
            list_of.update(dict.fromkeys(group.trigger_anchors, (revision.id, index)))
    groups: dict[SourceAnchor | tuple[str, int], list[EvidenceFragment]] = {}
    for fragment in catalog.fragments:
        groups.setdefault(list_of.get(fragment.anchor, fragment.anchor), []).append(fragment)
    return tuple(tuple(group) for group in groups.values())
