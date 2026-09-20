"""Shared revision input selection and baseline contracts."""

from dataclasses import replace

import pytest

from memforge.pipeline.projection_fragments import SupportRevalidationLimitation
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.revision_input import (
    ExtractionInputTask,
    InputCost,
    PlannedTransport,
    RevisionInputMode,
    RevisionInputPlanner,
    SupportInputTask,
)
from tests.test_revision_assessment import old_support, revisions


class CostPolicy:
    def __init__(self, *, delta=100, full=100, full_feasible=True):
        self.costs = {RevisionInputMode.DELTA: delta, RevisionInputMode.FULL: full}
        self.full_feasible = full_feasible

    def _cost(self, candidate):
        if candidate.mode is RevisionInputMode.FULL and not self.full_feasible:
            return None
        total = self.costs[candidate.mode]
        return InputCost(total - 10, 10, 1)

    def lower_bound(self, candidate):
        return self._cost(candidate)

    def materialize(self, candidate):
        cost = self._cost(candidate)
        return PlannedTransport(cost, candidate.mode.value) if cost is not None else None


def context_pair(
    base_text: str = "Existing policy.\n",
    current_text: str = "Existing policy.\n\nNew approval rule.\n",
):
    base, current = revisions(
        base_text,
        current_text,
    )
    return base, current, RevisionAssessmentContext(
        projection=current, base=base, access_context_hash="scope"
    )


def extraction_task(context, base):
    changed = next(
        fragment for fragment in context.full_fragments if "New approval" in fragment.presentation_text
    )
    return ExtractionInputTask(
        context.catalog((changed,)),
        named_baseline_revision_id=base.source_unit_revisions[0].id,
    )


def test_equal_total_cost_selects_delta():
    base, _current, context = context_pair()
    plan = RevisionInputPlanner.plan(
        context=context,
        task=extraction_task(context, base),
        request_policy=CostPolicy(delta=100, full=100),
    )
    assert plan.mode is RevisionInputMode.DELTA
    assert plan.selection_reason == "delta_cost_not_greater_than_full_lower_bound"


def test_strictly_cheaper_full_is_selected():
    base, _current, context = context_pair()
    plan = RevisionInputPlanner.plan(
        context=context,
        task=extraction_task(context, base),
        request_policy=CostPolicy(delta=101, full=100),
    )
    assert plan.mode is RevisionInputMode.FULL
    assert plan.selection_reason == "full_has_lower_total_request_cost"


def test_ineligible_full_falls_back_to_complete_delta():
    base, _current, context = context_pair()
    plan = RevisionInputPlanner.plan(
        context=context,
        task=extraction_task(context, base),
        request_policy=CostPolicy(delta=100, full=1, full_feasible=False),
    )
    assert plan.mode is RevisionInputMode.DELTA
    assert plan.estimated_cost.total_tokens == 100


def test_named_extraction_baseline_must_match_exactly():
    base, _current, context = context_pair()
    task = extraction_task(context, base)
    assert RevisionInputPlanner.plan(
        context=context, task=task, request_policy=CostPolicy()
    ).mode is RevisionInputMode.DELTA
    with pytest.raises(SupportRevalidationLimitation, match="named incremental baseline"):
        RevisionInputPlanner.plan(
            context=context,
            task=replace(task, named_baseline_revision_id="different-baseline"),
            request_policy=CostPolicy(),
        )


def test_unnamed_support_without_baseline_uses_full_but_named_missing_fails():
    base, current, _context = context_pair(
        base_text="Two reviewers approve US releases.\n",
        current_text="Two reviewers approve US releases.\n\nNew approval rule.\n",
    )
    no_baseline = RevisionAssessmentContext(
        projection=current, base=None, access_context_hash="scope"
    )
    support = old_support(base)
    plan = RevisionInputPlanner.plan(
        context=no_baseline,
        task=SupportInputTask((support,)),
        request_policy=CostPolicy(),
    )
    assert plan.mode is RevisionInputMode.FULL
    assert plan.selection_reason == "full_required_without_applicable_baseline"

    named = tuple(
        replace(part, validation_unit_revision_id=base.source_unit_revisions[0].id)
        for part in support
    )
    with pytest.raises(SupportRevalidationLimitation, match="named Support baseline"):
        RevisionInputPlanner.plan(
            context=no_baseline,
            task=SupportInputTask((named,)),
            request_policy=CostPolicy(),
        )
