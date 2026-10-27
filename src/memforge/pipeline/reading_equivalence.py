"""What changed between two revisions of one Observation, judged by what a model reads.

A Fragment's own reading is what it shows by itself: the canonical field it
belongs to, its kind and Fragment type, whether that field is selectable,
framing or other context, the field's descriptor label and the Fragment's core
presentation. Its reading adds what governs that meaning: the framing shown
with it that governs interpretation, the own readings of the anchors that
govern it (its headings, a list's lead-in, a declared text group's context and
its record's governing contextual fields) and its record schema's model
interpretation.

This one relation decides what changed wherever incremental work depends on it:

- Claim Extraction may cite as new Primary Evidence the Fragments whose reading
  changed;
- a Support assessment reads the Fragments whose reading changed as the
  revision's changes;
- a prior Evidence part corresponds exactly to the current Fragment with the
  same own reading. When only what governs it changed, the part is unchanged
  and the change is read with it. A part without a counterpart always has its
  replacement authorized, since its own reading is part of the reading.

Each consumer applies its own rules (multiplicity, uniqueness, eligibility) to
the same readings.

Markdown and plain text are also compared by their complete structures: every
Fragment of a structure that changed is changed, and every Fragment of a base
structure the target no longer has is removed.

Each revision is compiled under its own pinned profile. Whether a contextual
field or framing line governs interpretation is decided by the target
revision's declarations, so a representation upgrade that only declares a
field displayed is no change, and a field the target does not declare governs.
A displayed-only field has no reading of its own. Raw bytes and source origins
the reading never shows do not take part, and a tombstoned record has no
reading.

Two revisions with the same content under the same profile read the same,
whatever the profile, so a binary Artifact whose bytes are unchanged changes
nothing. Otherwise the whole target reads as changed when the comparison
cannot say less: its profile is outside this relation (a binary Artifact), its
structures cannot be compared with a base of another profile, or its base
cannot be compiled under its own older profile. A record with nothing to read,
such as a tombstoned one, changes nothing whatever its base. A target, or a
base under the target's profile, that cannot be compiled is
``ReadingUnavailable``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Literal

from memforge.pipeline.evidence_fragments import (
    EvidenceFragment,
    build_revision_fragment_index,
    canonical_record_is_tombstoned,
    changed_material_keys,
    revision_changed_structural_ranges,
)
from memforge.pipeline.revision_reading import RevisionReadingIndex, build_revision_reading_index
from memforge.source_projection import SourceAnchor, SourceObservationRevision
from memforge.source_representation import CanonicalRecordSchema, representation_contract_for_profile

# A comparable reading of one Fragment, equal across revisions and profiles when it reads the same.
ReadingKey = tuple[object, ...]
Range = tuple[int, int]

_RANGE_ADDRESSABLE_PROFILES = frozenset({"markdown-structural", "plain-text"})
_COMPARED_PROFILES = frozenset({"canonical-record", *_RANGE_ADDRESSABLE_PROFILES})


@dataclass(frozen=True, slots=True)
class _Reading:
    own: ReadingKey
    key: ReadingKey


@dataclass(frozen=True, slots=True)
class _Side:
    """One compiled revision: every Fragment anchor in order, and the readings among them."""

    revision: SourceObservationRevision
    anchors: tuple[SourceAnchor, ...]
    readings: dict[SourceAnchor, _Reading]
    counts: Counter[ReadingKey]


@dataclass(frozen=True, slots=True)
class _Changes:
    """The anchors of one side whose reading the other side lacks, and that side's changed structures.

    ``order`` is every Fragment anchor of that side, in order.
    """

    order: tuple[SourceAnchor, ...]
    read: frozenset[SourceAnchor]
    structures: tuple[Range, ...]

    def anchors(self) -> tuple[SourceAnchor, ...]:
        return tuple(
            anchor for anchor in self.order
            if anchor in self.read or any(_overlaps(anchor, structure) for structure in self.structures)
        )


_NO_CHANGE = _Changes((), frozenset(), ())


class ReadingUnavailable(ValueError):
    """One revision of a comparison cannot be compiled completely under the profile it is compared in."""

    def __init__(self, side: Literal["base", "target"], message: str) -> None:
        super().__init__(f"{side} revision: {message}")


class ReadingComparison:
    """The readings of a base and a target revision of one Observation.

    Each revision is compiled on first use. ``changed``, ``removed`` and
    ``changed_ranges`` return ``None`` when the whole Observation reads as
    changed, which includes a base that cannot be compiled under its own older
    profile. A method that compiles the revisions raises ``ReadingUnavailable``
    when the target cannot be compiled, or the base cannot be compiled under
    the profile it shares with the target.
    """

    def __init__(
        self,
        base: SourceObservationRevision,
        target: SourceObservationRevision,
        reading_index: Callable[[SourceObservationRevision], RevisionReadingIndex],
    ) -> None:
        self._base = base
        self._target = target
        self._reading_index = reading_index
        self._governs = _governs(_schema(target))

    def changed(self, *, purpose: Literal["authority", "current"] = "authority") -> tuple[SourceAnchor, ...] | None:
        """The target anchors that read differently from the base, in target order.

        Every Fragment of a changed Markdown or plain-text structure counts as
        changed. For ``authority``, a reading changed when no base Fragment reads the
        same: one more copy of unchanged text is no new assertion. For the
        ``current`` reading, every copy counts as changed when the number of
        copies changed.
        """

        changes = self._changes(purpose)
        return changes.anchors() if changes is not None else None

    def removed(self) -> tuple[SourceAnchor, ...] | None:
        """The base anchors the target no longer reads, in base order."""

        changes = self._changes("removed")
        return changes.anchors() if changes is not None else None

    def changed_ranges(self) -> tuple[Range, ...] | None:
        """The target ranges that may hold new assertions: changed structures and changed Fragments."""

        changes = self._changes("authority")
        if changes is None:
            return None
        return tuple(dict.fromkeys((*changes.structures, *(
            (anchor.range_start, anchor.range_end) for anchor in changes.order
            if anchor in changes.read and anchor.range_start is not None and anchor.range_end is not None
        ))))

    def counterparts(self, base_anchor: SourceAnchor) -> tuple[tuple[SourceAnchor, ...], tuple[SourceAnchor, ...]]:
        """Every base and every target anchor with the same own reading as one base anchor.

        A base anchor without an own reading, such as a displayed-only field,
        has no counterparts.
        """

        base, target = self._base_side, self._target_side
        reading = base.readings.get(base_anchor) if base is not None else None
        if reading is None:
            return (), ()
        return (
            tuple(anchor for anchor, other in base.readings.items() if other.own == reading.own),
            tuple(anchor for anchor, other in target.readings.items() if other.own == reading.own),
        )

    def _changes(self, purpose: Literal["authority", "current", "removed"]) -> _Changes | None:
        profile = self._target.evidence_profile
        if profile is None:
            return None
        if self._base.evidence_profile == profile and self._base.semantic_hash == self._target.semantic_hash:
            return _NO_CHANGE
        if profile.name not in _COMPARED_PROFILES:
            return None
        structured = profile.name in _RANGE_ADDRESSABLE_PROFILES
        if structured and self._base.evidence_profile != profile:
            return None
        target = self._target_side
        if purpose != "removed" and not structured and not target.readings:
            return _NO_CHANGE
        base = self._base_side
        if base is None:
            return None
        old, new = (target, base) if purpose == "removed" else (base, target)
        keys = changed_material_keys(old.counts, new.counts, purpose=purpose)
        return _Changes(
            new.anchors,
            frozenset(anchor for anchor, reading in new.readings.items() if reading.key in keys),
            revision_changed_structural_ranges(old.revision, new.revision, purpose=purpose) if structured else (),
        )

    @cached_property
    def _target_side(self) -> _Side:
        try:
            return _compile(self._target, self._reading_index, self._governs)
        except ValueError as error:
            raise ReadingUnavailable("target", str(error)) from error

    @cached_property
    def _base_side(self) -> _Side | None:
        """The compiled base; ``None`` when it cannot be compiled under its own older profile."""

        try:
            return _compile(self._base, self._reading_index, self._governs)
        except ValueError as error:
            if self._base.evidence_profile != self._target.evidence_profile:
                return None
            raise ReadingUnavailable("base", str(error)) from error


def compare_readings(
    base: SourceObservationRevision,
    target: SourceObservationRevision,
    *,
    reading_index: Callable[[SourceObservationRevision], RevisionReadingIndex] | None = None,
) -> ReadingComparison:
    """Compare two revisions of one Observation by what a model reads in each.

    ``reading_index`` supplies a revision's complete reading index, so that a
    caller that already holds indexes reuses them and decides how a revision
    that cannot be compiled fails; by default each revision is compiled here
    and such a revision raises ``ValueError``.
    """

    if base.observation_id != target.observation_id:
        raise ValueError("reading comparison requires two revisions of one Observation")
    return ReadingComparison(base, target, reading_index or _complete_reading_index)


def _complete_reading_index(revision: SourceObservationRevision) -> RevisionReadingIndex:
    fragments = build_revision_fragment_index(revision)
    if not fragments.complete:
        raise ValueError("revision cannot be compiled completely")
    return build_revision_reading_index(revision, fragments.fragments)


def _schema(revision: SourceObservationRevision) -> CanonicalRecordSchema | None:
    contract = representation_contract_for_profile(revision.evidence_profile)
    return contract.canonical_schema if contract is not None else None


def _governs(schema: CanonicalRecordSchema | None) -> Callable[[str | None], bool]:
    """Whether a pointer governs interpretation under one schema; a pointer it does not declare does."""

    displayed = tuple(
        field for field in (schema.fields if schema is not None else ()) if not field.governs_interpretation
    )

    def governs(pointer: str | None) -> bool:
        return pointer is None or not any(field.names(pointer) for field in displayed)

    return governs


def _compile(
    revision: SourceObservationRevision,
    reading_index: Callable[[SourceObservationRevision], RevisionReadingIndex],
    governs: Callable[[str | None], bool],
) -> _Side:
    schema = _schema(revision)
    if schema is not None and canonical_record_is_tombstoned(revision):
        return _Side(revision, (), {}, Counter())
    index = reading_index(revision)
    by_anchor = {fragment.anchor: fragment for fragment in index.fragments}

    def displayed_only(fragment: EvidenceFragment) -> bool:
        return fragment.reading.role != "selectable" and not governs(fragment.reading.field)

    governing: dict[SourceAnchor, set[ReadingKey]] = {anchor: set() for anchor in by_anchor}
    for group in index.groups:
        context = {
            anchor: _own_reading(by_anchor[anchor]) for anchor in group.governing_anchors
            if not displayed_only(by_anchor[anchor])
        }
        for trigger in group.trigger_anchors:
            governing[trigger].update(own for anchor, own in context.items() if anchor != trigger)
    interpretation = schema.model_interpretation if schema is not None else None
    readings = {}
    for fragment in index.fragments:
        if displayed_only(fragment):
            continue
        own = _own_reading(fragment)
        readings[fragment.anchor] = _Reading(own, (
            *own,
            tuple((label, value) for pointer, label, value in fragment.reading.framing if governs(pointer)),
            tuple(sorted(governing[fragment.anchor], key=json.dumps)),
            interpretation,
        ))
    return _Side(revision, tuple(by_anchor), readings, Counter(reading.key for reading in readings.values()))


def _own_reading(fragment: EvidenceFragment) -> ReadingKey:
    """What a Fragment shows by itself, without the context that governs it."""

    reading = fragment.reading
    return fragment.kind.value, fragment.fragment_type, reading.field, reading.role, reading.label, reading.core


def _overlaps(anchor: SourceAnchor, structure: Range) -> bool:
    start, end = anchor.range_start, anchor.range_end
    return start is not None and end is not None and start < structure[1] and structure[0] < end
