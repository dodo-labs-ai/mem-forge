"""Exact prior Evidence correspondence, whole-Support routing, ChangeBundle and reading order for one revision.

Everything here is program-only and recomputed for every base/target pair from
the persisted Evidence digests, the target membership, its coverage and its
reading structure. No correspondence status is stored, no semantic similarity
is consulted, and no cost comparison decides what a Support reads.

A revision's changes are its changed current ReadingGroups followed by the old
text of what it removed. A list that only lost items is a changed group too.
They are the ChangeBundle that Change Impact judges an exactly unchanged Support
against, and the start of every reading order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from enum import Enum
from typing import Any

from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole, ResolvedEvidenceSelection
from memforge.models import Memory
from memforge.pipeline.evidence_fragments import EvidenceFragment
from memforge.pipeline.projection_fragments import (
    ProjectionFragmentCatalog,
    SupportRevalidationLimitation,
    SupportRevalidationLimitationCode,
)
from memforge.pipeline.revision_assessment import RevisionAssessmentContext, reading_group_label
from memforge.source_artifacts import source_artifact_content_sha256
from memforge.source_projection import AnchorKind, EvidenceCoordinateSpace, ProjectionCoverage, SourceAnchor


# One Evidence part's exact identity: its Observation, raw digest and presentation digest.
EvidenceDigest = tuple[str, str, str]


def evidence_digests(selection: ResolvedEvidenceSelection) -> tuple[EvidenceDigest, ...]:
    """The exact identities of a resolved selection's parts, sorted and distinct."""
    return tuple(sorted({
        (part.anchor.observation_id, part.raw_content_sha256, part.presentation_sha256) for part in selection.parts
    }))


def exact_evidence_selection(
    context: RevisionAssessmentContext, parts: Sequence[ActiveSupportEvidence],
) -> ResolvedEvidenceSelection | None:
    """Use the same pinned, bilateral correspondence for a staged selection."""
    catalog = context.catalog(context.full_fragments)
    refs: dict[EvidenceRole, list[str]] = {EvidenceRole.PRIMARY: [], EvidenceRole.REQUIRED: []}
    displays = {}
    for correspondence in correspond_evidence(context, parts):
        if correspondence.status is not EvidenceCorrespondence.EXACT_UNCHANGED:
            return None
        refs[correspondence.evidence.role].append(correspondence.current[0].reference)
        if correspondence.evidence.presentation_sha256 == correspondence.current[0].presentation_sha256:
            displays[correspondence.current[0].reference] = correspondence.evidence.display_text
    if len(refs[EvidenceRole.PRIMARY]) != 1:
        return None
    primary_ref = refs[EvidenceRole.PRIMARY][0]
    required = tuple(dict.fromkeys(ref for ref in refs[EvidenceRole.REQUIRED] if ref != primary_ref))
    return catalog.resolve_selection(primary_ref=primary_ref, required_refs=required, display_text_by_ref=displays)


def correspond_evidence(context, parts):
    """Correspond pinned parts against the complete operation-local indexes."""
    catalog = context.catalog(context.full_fragments)
    by_observation: dict[str, dict[SourceAnchor, EvidenceFragment]] = {}
    for fragment in catalog.fragments:
        by_observation.setdefault(fragment.anchor.observation_id, {})[fragment.anchor] = fragment
    returned = frozenset(observation.id for observation in context.projection.observations)
    return tuple(_correspond(part, context, returned, by_observation) for part in parts)


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
    CHANGE_IMPACT = "change_impact"
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
        return reading_group_label(self.fragments)


@dataclass(frozen=True)
class SupportReadingOrder:
    """One fixed order over the complete current revision for every assessed Support of a cohort."""

    parts: tuple[ReadingPart, ...]
    # Work id -> how many leading parts form that Support's first part.
    first_part_end: Mapping[str, int]

    @property
    def removed(self) -> tuple[Mapping[str, Any], ...]:
        return removed_entries(self.parts)

    @property
    def has_current_content(self) -> bool:
        return any(part.fragments for part in self.parts)


@dataclass(frozen=True)
class SupportRevisionPlan:
    context: RevisionAssessmentContext
    # The complete current Support catalog; every current ref here belongs to it.
    catalog: ProjectionFragmentCatalog
    supports: tuple[SupportPlan, ...]
    # Current ReadingGroups in document order: one outermost list, or one Fragment, each.
    groups: tuple[tuple[EvidenceFragment, ...], ...]
    # What this revision changed: the changed current ReadingGroups in document order,
    # then the old text of each removed Fragment. Empty without a usable baseline.
    changes: tuple[ReadingPart, ...]
    # Catalog refs of the added or modified current Fragments inside ``changes``;
    # the other Fragments of a changed group are unchanged context.
    changed_refs: frozenset[str]

    def reading_order(self, assessed: Sequence[SupportPlan]) -> SupportReadingOrder:
        """Changes and the assessed Supports' own prior Evidence first; then the rest, in document order."""
        if self.context.base is None:
            # Nothing is known to be unchanged, so every Support's first part is the whole revision.
            parts = tuple(ReadingPart(fragments=group) for group in self.groups)
            return SupportReadingOrder(parts, {support.item.id: len(parts) for support in assessed})
        own_anchors = {
            fragment.anchor for support in assessed for correspondence in support.parts for fragment in correspondence.current
        }
        changed_anchors = {fragment.anchor for part in self.changes for fragment in part.fragments}
        unchanged = [group for group in self.groups if not any(f.anchor in changed_anchors for f in group)]
        own = [group for group in unchanged if any(f.anchor in own_anchors for f in group)]
        rest = [group for group in unchanged if not any(f.anchor in own_anchors for f in group)]
        parts = (*self.changes, *(ReadingPart(fragments=group) for group in (*own, *rest)))
        position = {
            fragment.anchor: index for index, part in enumerate(parts) for fragment in part.fragments
        }

        def first_part_end(support: SupportPlan) -> int:
            own_ends = (
                position[fragment.anchor] + 1 for correspondence in support.parts for fragment in correspondence.current
            )
            # Prior Evidence travels with the first part, so every Support reads at least one part before concluding.
            return max((1, len(self.changes), *own_ends))

        return SupportReadingOrder(parts, {support.item.id: first_part_end(support) for support in assessed})

    def evidence_reading_order(
        self, assessed: Sequence[SupportPlan], evidence: Sequence[EvidenceDigest],
    ) -> SupportReadingOrder:
        """Only the current ReadingGroups that hold the given Evidence, in document order, read as one first part."""
        wanted = set(evidence)
        groups = self.groups or self.context.reading_groups(self.catalog.fragments)
        parts = tuple(
            ReadingPart(fragments=group) for group in groups if any(_digests(fragment) in wanted for fragment in group)
        )
        return SupportReadingOrder(parts, {support.item.id: len(parts) for support in assessed})


def plan_support_revision(
    context: RevisionAssessmentContext, items: Sequence[SupportWorkItem]
) -> SupportRevisionPlan:
    """Correspond and route every fixed Support that shares one base/target pair."""
    catalog = context.catalog(context.full_fragments)
    by_observation: dict[str, dict[SourceAnchor, EvidenceFragment]] = {}
    for fragment in catalog.fragments:
        by_observation.setdefault(fragment.anchor.observation_id, {})[fragment.anchor] = fragment
    returned = frozenset(observation.id for observation in context.projection.observations)
    # Unavailable retained interpretation remains UNKNOWN for the whole Support;
    # it is never dropped to make the remaining parts exactly reusable.
    correspondences = [
        tuple(
            _correspond(part, context, returned, by_observation)
            for part in item.support
        )
        for item in items
    ]
    # Only a Support with an UNKNOWN part neither judges nor reads the revision.
    reads_revision = any(not _any_unknown(parts) for parts in correspondences)
    groups = context.reading_groups(catalog.fragments) if reads_revision else ()
    changes, changed_refs = (
        _changes(context, catalog, groups) if context.base is not None and reads_revision else ((), frozenset())
    )
    supports = tuple(
        SupportPlan(item, parts, _route(parts, context, bool(changes)))
        for item, parts in zip(items, correspondences, strict=True)
    )
    return SupportRevisionPlan(context, catalog, supports, groups, changes, changed_refs)


def _correspond(
    part: ActiveSupportEvidence,
    context: RevisionAssessmentContext,
    returned: frozenset[str],
    by_observation: dict[str, dict[SourceAnchor, EvidenceFragment]],
) -> PartCorrespondence:
    old_revision = context.revisions.get(part.anchor.observation_revision_id)
    if old_revision is not None:
        _verify_pinned_integrity(part, old_revision)
    if (part.anchor.observation_revision_id in context.retired
            or old_revision is not None and old_revision.evidence_profile is None
            or part.text_view is not None and part.text_view.version != 1):
        return PartCorrespondence(part, EvidenceCorrespondence.UNKNOWN)
    old_selection = ()
    old_material = {}
    if old_revision is not None:
        old_anchors, old_material = context.selection_index(old_revision)
        old_selection = old_anchors.get(part.anchor, ())
        if part.text_view is not None and (len(old_selection) != 1 or part.text_view != old_selection[0].text_view):
            raise SupportRevalidationLimitation(
                SupportRevalidationLimitationCode.EVIDENCE_INTEGRITY,
                "Stored text view does not reconstruct from its pinned source selection",
            )
        if len(old_selection) == 1 and part.raw_content_sha256 is not None:
            old_fragment = old_selection[0]
            if (
                (part.text_view is not None and part.excerpt != old_fragment.presentation_text)
                or old_fragment.raw_content_sha256 != part.raw_content_sha256
                or part.presentation_sha256 is not None
                and old_fragment.presentation_sha256 != part.presentation_sha256
            ):
                raise SupportRevalidationLimitation(
                    SupportRevalidationLimitationCode.EVIDENCE_INTEGRITY,
                    "Stored Evidence integrity differs from its pinned source selection",
                )
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
    if old_revision is None:
        # Current complete source can still assess the fixed claim. Historical
        # authority missing here forbids exact reuse, not a fresh assessment.
        return PartCorrespondence(part, EvidenceCorrespondence.MODIFIED)
    if len(old_selection) == 1 and part.raw_content_sha256 is not None:
        old_fragment = old_selection[0]
        target_revision = context.current.get(observation_id)
        if target_revision is not None and old_revision.evidence_profile == target_revision.evidence_profile:
            current_by_anchor = by_observation.get(observation_id, {})
            if old_revision.id == target_revision.id:
                # An exact pinned occurrence needs no cross-version disambiguation.
                match = current_by_anchor.get(part.anchor)
                candidates = (match,) if match is not None else ()
                unique_old = True
            else:
                key = (old_fragment.kind, old_fragment.fragment_type, old_fragment.content_value)
                unique_old = len(old_material.get(key, ())) == 1
                _, target_material = context.selection_index(target_revision)
                matches = target_material.get(key, ())
                # Catalog refs are request-local; rebind indexed source anchors.
                candidates = tuple(current_by_anchor[fragment.anchor] for fragment in matches
                                   if fragment.anchor in current_by_anchor)
            if candidates and (not unique_old or len(candidates) > 1):
                return PartCorrespondence(part, EvidenceCorrespondence.AMBIGUOUS, candidates)
            if len(candidates) == 1 and (
                part.role is not EvidenceRole.PRIMARY or candidates[0].primary_eligible
            ):
                return PartCorrespondence(part, EvidenceCorrespondence.EXACT_UNCHANGED, candidates)
    # The Observation is still an authoritative member, but the exact text is gone. When
    # its revision is unchanged (a legacy part without digests, or a recompiled split),
    # the old anchor still locates the current text it is read with.
    overlapping = tuple(
        fragment for fragment in by_observation.get(observation_id, {}).values()
        if _overlaps(part.anchor, fragment.anchor)
    )
    return PartCorrespondence(part, EvidenceCorrespondence.MODIFIED, overlapping)


def _verify_pinned_integrity(part: ActiveSupportEvidence, revision) -> None:
    """Check retained bytes independently of current coverage or renderer availability."""
    try:
        if revision.observation_id != part.anchor.observation_id:
            raise ValueError("pinned Evidence Observation identity mismatch")
        if part.text_view is not None:
            part.text_view.verify_origins(part.anchor, revision.content)
        start, end = part.anchor.range_start, part.anchor.range_end
        profile = revision.evidence_profile
        if profile is not None and profile.coordinate_space is EvidenceCoordinateSpace.WHOLE_ARTIFACT:
            if (part.raw_content_sha256 is not None
                    and source_artifact_content_sha256(revision.metadata) != part.raw_content_sha256):
                raise ValueError("pinned Artifact byte integrity mismatch")
        elif part.anchor.kind is AnchorKind.WHOLE_OBSERVATION:
            start, end = 0, len(revision.content)
        if start is not None and end is not None and part.raw_content_sha256 is not None:
            if end > len(revision.content) or sha256(revision.content[start:end].encode()).hexdigest() != part.raw_content_sha256:
                raise ValueError("pinned Evidence raw integrity mismatch")
        if part.excerpt is not None and part.presentation_sha256 is not None:
            if sha256(part.excerpt.encode()).hexdigest() != part.presentation_sha256:
                raise ValueError("stored Evidence presentation integrity mismatch")
    except ValueError as error:
        raise SupportRevalidationLimitation(
            SupportRevalidationLimitationCode.EVIDENCE_INTEGRITY, str(error),
        ) from error


def _overlaps(old: SourceAnchor, current: SourceAnchor) -> bool:
    """Whether a current Fragment lies on an old anchor of the same Observation revision."""
    if current.observation_revision_id != old.observation_revision_id:
        return False
    if old.fragment_id is not None and current.fragment_id == old.fragment_id:
        return True
    ranges = (old.range_start, old.range_end, current.range_start, current.range_end)
    return None not in ranges and current.range_start < old.range_end and old.range_start < current.range_end


def _all_exact(parts: tuple[PartCorrespondence, ...]) -> bool:
    return all(correspondence.status is EvidenceCorrespondence.EXACT_UNCHANGED for correspondence in parts)


def _has_primary(parts: tuple[PartCorrespondence, ...]) -> bool:
    return any(correspondence.evidence.role is EvidenceRole.PRIMARY for correspondence in parts)


def _any_unknown(parts: tuple[PartCorrespondence, ...]) -> bool:
    return any(correspondence.status is EvidenceCorrespondence.UNKNOWN for correspondence in parts)


def _route(parts: tuple[PartCorrespondence, ...], context: RevisionAssessmentContext, changed: bool) -> SupportRoute:
    # An UNKNOWN part never reaches the model, so it decides the whole Support.
    if _any_unknown(parts):
        return SupportRoute.UNRESOLVED_PARTIAL_COVERAGE
    # Without a usable baseline nothing is known to be unchanged: read the whole revision.
    # A Support whose Primary part was dropped has nothing to rebind to.
    if context.base is None or not _all_exact(parts) or not _has_primary(parts):
        return SupportRoute.SUPPORT_ASSESSMENT
    return SupportRoute.CHANGE_IMPACT if changed else SupportRoute.REBIND_SUPPORT


def _changes(
    context: RevisionAssessmentContext,
    catalog: ProjectionFragmentCatalog,
    groups: tuple[tuple[EvidenceFragment, ...], ...],
) -> tuple[tuple[ReadingPart, ...], frozenset[str]]:
    """The changed current ReadingGroups, whole, then removed old text: it may have qualified a claim."""
    changed_fragments, removed = context.delta_fragments()
    changed_anchors = {fragment.anchor for fragment in changed_fragments}
    # A list that lost an item changed, so its remaining items are read with the removal.
    touched = changed_anchors | _remaining_list_items(context, catalog, removed)
    changed_groups = tuple(group for group in groups if any(f.anchor in touched for f in group))
    changes = (
        *(ReadingPart(fragments=group) for group in changed_groups),
        *(
            ReadingPart(removed={"ref": f"h{index:06d}", **context.removed_entry(fragment), **_headings(context, fragment)})
            for index, fragment in enumerate(removed)
        ),
    )
    changed_refs = frozenset(fragment.reference for fragment in catalog.fragments if fragment.anchor in changed_anchors)
    return changes, changed_refs


def _remaining_list_items(
    context: RevisionAssessmentContext, catalog: ProjectionFragmentCatalog, removed: tuple[EvidenceFragment, ...]
) -> set[SourceAnchor]:
    """Current anchors of the items still present from a baseline list that lost some of its items.

    An item is found by its exact digests in the same Observation, like prior Evidence.
    Identical text elsewhere in that Observation is read too; it only adds context.
    """
    removed_anchors = {fragment.anchor for fragment in removed}
    baselines = {fragment.anchor.observation_revision_id: context.previous[fragment.anchor.observation_id] for fragment in removed}
    remaining = set()
    for baseline in baselines.values():
        reading = context.reading_index(baseline)
        by_anchor = {fragment.anchor: fragment for fragment in reading.fragments}
        for group in reading.lists:
            if not removed_anchors.isdisjoint(group.trigger_anchors):
                remaining.update(_digests(by_anchor[anchor]) for anchor in group.trigger_anchors if anchor not in removed_anchors)
    return {fragment.anchor for fragment in catalog.fragments if _digests(fragment) in remaining}


def _digests(fragment: EvidenceFragment) -> tuple[str, str, str]:
    return fragment.anchor.observation_id, fragment.raw_content_sha256, fragment.presentation_sha256


def _headings(context: RevisionAssessmentContext, fragment: EvidenceFragment) -> dict:
    """The heading path removed Markdown text sat under in its baseline."""
    headings = context.heading_context(fragment)
    return {"heading_context": list(headings)} if headings else {}


def removed_entries(parts: Sequence[ReadingPart]) -> tuple[Mapping[str, Any], ...]:
    """The removed old text among some reading parts, in order."""
    return tuple(part.removed for part in parts if part.removed is not None)
