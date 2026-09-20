"""One provider-neutral policy for revision reading scope and input mode."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol

from memforge.pipeline.projection_fragments import (
    ProjectionFragmentCatalog,
    SupportRevalidationLimitation,
    SupportRevalidationLimitationCode,
)


class RevisionInputMode(str, Enum):
    DELTA = "delta"
    FULL = "full"


@dataclass(frozen=True)
class ExtractionInputTask:
    """L1 work whose supplied Primary authority must remain exact."""

    authorized: ProjectionFragmentCatalog
    named_baseline_revision_id: str | None = None


@dataclass(frozen=True)
class SupportInputTask:
    """L3 work over fixed claims and their independently validated Support."""

    supports: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class InputCost:
    """Complete cost of an executable request plan in provider token units."""

    input_tokens: int
    output_tokens: int
    request_count: int
    image_count: int = 0
    image_bytes: int = 0
    complete: bool = True

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class InputCandidate:
    mode: RevisionInputMode
    catalog: ProjectionFragmentCatalog
    reading_groups: tuple[Any, ...]
    reading_indexes: tuple[Any, ...] = ()
    removed_historical: tuple[dict[str, Any], ...] = ()
    include_history: bool = False


@dataclass(frozen=True)
class PlannedTransport:
    cost: InputCost
    payload: Any


@dataclass(frozen=True)
class InputPlan:
    mode: RevisionInputMode
    catalog: ProjectionFragmentCatalog
    reading_groups: tuple[Any, ...]
    removed_historical: tuple[dict[str, Any], ...]
    selection_reason: str
    estimated_cost: InputCost
    transport: Any
    include_history: bool = False


class RevisionRequestPolicy(Protocol):
    """Task-specific request formatter and capacity estimator."""

    def lower_bound(self, candidate: InputCandidate) -> InputCost | None: ...

    def materialize(self, candidate: InputCandidate) -> PlannedTransport | None: ...


class RevisionInputPlanner:
    """Construct complete delta/full candidates and choose the cheaper plan."""

    @staticmethod
    def _baseline_id(context) -> str | None:
        return context.base.source_unit_revisions[0].id if context.base is not None else None

    @classmethod
    def _validate_baseline(cls, context, task: ExtractionInputTask | SupportInputTask) -> None:
        actual = cls._baseline_id(context)
        if isinstance(task, ExtractionInputTask):
            expected = task.named_baseline_revision_id
            if expected is not None and actual != expected:
                raise SupportRevalidationLimitation(
                    SupportRevalidationLimitationCode.UNSUPPORTED_REPRESENTATION,
                    "named incremental baseline is unavailable or does not match the prepared revision",
                )
            return

        named = {
            part.validation_unit_revision_id
            for support in task.supports
            for part in support
            if part.validation_unit_revision_id is not None
        }
        if len(named) > 1 or (named and actual not in named):
            raise SupportRevalidationLimitation(
                SupportRevalidationLimitationCode.UNSUPPORTED_REPRESENTATION,
                "named Support baseline is unavailable or does not match the prepared revision",
            )

    @staticmethod
    def _reading_expansion(context, selected, *, added_primary_eligible: bool):
        from memforge.pipeline.revision_reading import build_revision_reading_index

        selected_by_anchor = {fragment.anchor: fragment for fragment in selected}
        expanded_by_anchor = dict(selected_by_anchor)
        reading_groups = []
        reading_indexes = []
        for revision in context.current.values():
            fragments = tuple(
                fragment
                for fragment in context.full_fragments
                if fragment.anchor.observation_revision_id == revision.id
            )
            authority = tuple(fragment for fragment in fragments if fragment.anchor in selected_by_anchor)
            if not authority:
                continue
            index = (
                context.reading_index(revision)
                if hasattr(context, "reading_index")
                else build_revision_reading_index(revision, fragments)
            )
            reading_indexes.append(index)
            expansion = index.expand(authority)
            for fragment in expansion.fragments:
                expanded_by_anchor[fragment.anchor] = (
                    selected_by_anchor[fragment.anchor]
                    if fragment.anchor in expansion.authority_anchors
                    else fragment
                    if added_primary_eligible
                    else replace(fragment, primary_eligible=False)
                )
            included = set(expansion.authority_anchors) | set(expansion.context_anchors)
            reading_groups.extend(
                group for group in index.groups if included.intersection(group.fragment_anchors)
            )
        ordered = tuple(
            expanded_by_anchor[fragment.anchor]
            for fragment in context.full_fragments
            if fragment.anchor in expanded_by_anchor
        )
        # Preserve any caller fragment outside current reading indexes so the
        # planner cannot silently narrow already-authorized work.
        ordered += tuple(
            fragment for anchor, fragment in expanded_by_anchor.items()
            if anchor not in {item.anchor for item in ordered}
        )
        return ordered, tuple(reading_groups), tuple(reading_indexes)

    @classmethod
    def _extraction_candidates(cls, context, task: ExtractionInputTask):
        authorized = tuple(task.authorized.fragments)
        delta_fragments, delta_groups, delta_indexes = cls._reading_expansion(
            context, authorized, added_primary_eligible=False
        )
        delta = InputCandidate(
            RevisionInputMode.DELTA,
            context.catalog(delta_fragments),
            delta_groups,
            delta_indexes,
        )

        authorized_by_anchor = {fragment.anchor: fragment for fragment in authorized}
        full_fragments = tuple(
            authorized_by_anchor.get(
                fragment.anchor,
                replace(fragment, primary_eligible=False),
            )
            for fragment in context.full_fragments
        )
        full_expanded, full_groups, full_indexes = cls._reading_expansion(
            context, full_fragments, added_primary_eligible=False
        )
        full = InputCandidate(
            RevisionInputMode.FULL,
            context.catalog(full_expanded),
            full_groups,
            full_indexes,
        )
        return (full,) if context.base is None else (delta, full)

    @classmethod
    def _support_candidates(cls, context, task: SupportInputTask):
        full_fragments, full_groups, full_indexes = cls._reading_expansion(
            context, context.full_fragments, added_primary_eligible=True
        )
        full = InputCandidate(
            RevisionInputMode.FULL,
            context.catalog(full_fragments),
            full_groups,
            full_indexes,
            include_history=False,
        )
        if context.base is None:
            return (full,)

        changed, removed = context.delta()
        selected = {fragment.anchor: fragment for fragment in changed}
        for support in task.supports:
            for part in support:
                for fragment in context.full_fragments:
                    if fragment.anchor.observation_id == part.anchor.observation_id and (
                        fragment.anchor == part.anchor or fragment.presentation_text == part.excerpt
                    ):
                        selected[fragment.anchor] = fragment
        delta_fragments, delta_groups, delta_indexes = cls._reading_expansion(
            context, tuple(selected.values()), added_primary_eligible=True
        )
        delta = InputCandidate(
            RevisionInputMode.DELTA,
            context.catalog(delta_fragments),
            delta_groups,
            delta_indexes,
            tuple(removed),
            include_history=True,
        )
        return delta, full

    @staticmethod
    def _result(candidate: InputCandidate, transport: PlannedTransport, reason: str) -> InputPlan:
        return InputPlan(
            mode=candidate.mode,
            catalog=candidate.catalog,
            reading_groups=candidate.reading_groups,
            removed_historical=candidate.removed_historical,
            selection_reason=reason,
            estimated_cost=transport.cost,
            transport=transport.payload,
            include_history=candidate.include_history,
        )

    @classmethod
    def plan(
        cls,
        *,
        context,
        task: ExtractionInputTask | SupportInputTask,
        request_policy: RevisionRequestPolicy,
    ) -> InputPlan:
        cls._validate_baseline(context, task)
        candidates = (
            cls._extraction_candidates(context, task)
            if isinstance(task, ExtractionInputTask)
            else cls._support_candidates(context, task)
        )
        by_mode = {candidate.mode: candidate for candidate in candidates}
        delta = by_mode.get(RevisionInputMode.DELTA)
        full = by_mode[RevisionInputMode.FULL]

        if delta is None:
            planned = request_policy.materialize(full)
            if planned is None:
                raise SupportRevalidationLimitation(
                    SupportRevalidationLimitationCode.CAPACITY_EXCEEDED,
                    "complete current revision input exceeds configured capability",
                )
            return cls._result(full, planned, "full_required_without_applicable_baseline")

        planned_delta = request_policy.materialize(delta)
        full_lower_bound = request_policy.lower_bound(full)
        if planned_delta is not None and (
            full_lower_bound is None
            or full_lower_bound.total_tokens >= planned_delta.cost.total_tokens
        ):
            return cls._result(delta, planned_delta, "delta_cost_not_greater_than_full_lower_bound")

        planned_full = request_policy.materialize(full)
        if planned_full is None:
            if planned_delta is not None:
                return cls._result(delta, planned_delta, "full_exceeds_capacity")
            raise SupportRevalidationLimitation(
                SupportRevalidationLimitationCode.CAPACITY_EXCEEDED,
                "neither complete delta nor complete current revision fits configured capability",
            )
        if planned_delta is None or planned_full.cost.total_tokens < planned_delta.cost.total_tokens:
            return cls._result(full, planned_full, "full_has_lower_total_request_cost")
        return cls._result(delta, planned_delta, "delta_wins_equal_total_request_cost")
