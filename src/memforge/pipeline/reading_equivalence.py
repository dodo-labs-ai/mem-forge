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

Each revision is compiled under its own pinned profile. Whether a contextual
field or framing line governs interpretation is decided by the target
revision's declarations, so a representation upgrade that only declares a
field displayed is no change, and a field the target does not declare governs.
A displayed-only field has no reading of its own. Raw bytes and source origins
the reading never shows do not take part, and a tombstoned record has no
reading.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal, Mapping

from memforge.pipeline.evidence_fragments import (
    EvidenceFragment,
    _changed_material_keys,
    build_revision_fragment_index,
    canonical_record_is_tombstoned,
)
from memforge.pipeline.revision_reading import RevisionReadingIndex, build_revision_reading_index
from memforge.source_projection import SourceAnchor, SourceObservationRevision
from memforge.source_representation import CanonicalRecordSchema, representation_contract_for_profile

# A comparable reading of one Fragment, equal across revisions and profiles when it reads the same.
ReadingKey = tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _Reading:
    own: ReadingKey
    key: ReadingKey


class ReadingUnavailable(ValueError):
    """One revision of a comparison cannot be compiled completely under its pinned profile."""

    def __init__(self, side: Literal["base", "target"], message: str) -> None:
        super().__init__(f"{side} revision: {message}")
        self.side = side


class ReadingComparison:
    """The readings of a base and a target revision of one Observation."""

    __slots__ = ("_base", "_target", "_base_counts", "_target_counts")

    def __init__(self, base: Mapping[SourceAnchor, _Reading], target: Mapping[SourceAnchor, _Reading]) -> None:
        self._base = dict(base)
        self._target = dict(target)
        self._base_counts = Counter(reading.key for reading in self._base.values())
        self._target_counts = Counter(reading.key for reading in self._target.values())

    def changed(self, *, purpose: Literal["authority", "current"] = "authority") -> tuple[SourceAnchor, ...]:
        """The target anchors whose reading changed, in target order.

        For ``authority``, a reading changed when no base Fragment reads the
        same: one more copy of unchanged text is no new assertion. For the
        ``current`` reading, every copy counts as changed when the number of
        copies changed.
        """

        changed = _changed_material_keys(self._base_counts, self._target_counts, purpose=purpose)
        return tuple(anchor for anchor, reading in self._target.items() if reading.key in changed)

    def removed(self) -> tuple[SourceAnchor, ...]:
        """The base anchors whose reading no target Fragment has, in base order."""

        removed = _changed_material_keys(self._target_counts, self._base_counts, purpose="removed")
        return tuple(anchor for anchor, reading in self._base.items() if reading.key in removed)

    def counterparts(self, base_anchor: SourceAnchor) -> tuple[tuple[SourceAnchor, ...], tuple[SourceAnchor, ...]]:
        """Every base and every target anchor with the same own reading as one base anchor.

        A base anchor without an own reading, such as a displayed-only field,
        has no counterparts.
        """

        reading = self._base.get(base_anchor)
        if reading is None:
            return (), ()
        return (
            tuple(anchor for anchor, other in self._base.items() if other.own == reading.own),
            tuple(anchor for anchor, other in self._target.items() if other.own == reading.own),
        )


def compare_readings(
    base: SourceObservationRevision,
    target: SourceObservationRevision,
    *,
    reading_index: Callable[[SourceObservationRevision], RevisionReadingIndex] | None = None,
) -> ReadingComparison:
    """Compare two revisions of one Observation by what a model reads in each.

    ``reading_index`` supplies a revision's complete reading index, so that a
    caller that already holds indexes reuses them; by default each revision is
    compiled here. Raises ``ReadingUnavailable`` naming the side that cannot be
    compiled completely under its pinned profile.
    """

    if base.observation_id != target.observation_id:
        raise ValueError("reading comparison requires two revisions of one Observation")
    index = reading_index or _complete_reading_index
    governs = _governs(_schema(target))
    return ReadingComparison(
        _readings(base, index, governs, side="base"),
        _readings(target, index, governs, side="target"),
    )


def _complete_reading_index(revision: SourceObservationRevision) -> RevisionReadingIndex:
    fragments = build_revision_fragment_index(revision)
    if not fragments.complete:
        raise ValueError("revision cannot be compiled completely")
    return build_revision_reading_index(revision, fragments.fragments)


def _schema(revision: SourceObservationRevision) -> CanonicalRecordSchema | None:
    contract = representation_contract_for_profile(revision.evidence_profile)
    return contract.canonical_schema if contract is not None else None


def _governs(schema: CanonicalRecordSchema | None) -> Callable[[str | None], bool]:
    """Whether a declared pointer governs interpretation under one schema; undeclared pointers do."""

    displayed = tuple(
        field.json_pointer.split("/") for field in (schema.fields if schema is not None else ())
        if not field.governs_interpretation
    )

    def governs(pointer: str | None) -> bool:
        if pointer is None:
            return True
        parts = pointer.split("/")
        return not any(
            len(pattern) == len(parts)
            and all(
                expected == value or expected == "*" and (value == "*" or value.isdecimal())
                for expected, value in zip(pattern, parts, strict=True)
            )
            for pattern in displayed
        )

    return governs


def _readings(
    revision: SourceObservationRevision,
    reading_index: Callable[[SourceObservationRevision], RevisionReadingIndex],
    governs: Callable[[str | None], bool],
    *,
    side: Literal["base", "target"],
) -> dict[SourceAnchor, _Reading]:
    schema = _schema(revision)
    try:
        if schema is not None and canonical_record_is_tombstoned(revision):
            return {}
        index = reading_index(revision)
    except ValueError as error:
        raise ReadingUnavailable(side, str(error)) from error
    if any(fragment.reading is None for fragment in index.fragments):
        raise ReadingUnavailable(side, "a Fragment was compiled without its reading")
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
    return readings


def _own_reading(fragment: EvidenceFragment) -> ReadingKey:
    """What a Fragment shows by itself, without the context that governs it."""

    reading = fragment.reading
    return fragment.kind.value, fragment.fragment_type, reading.field, reading.role, reading.label, reading.core
