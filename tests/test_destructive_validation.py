"""DestructiveValidation keeps any destructive decision whose recorded facts are incomplete."""

from __future__ import annotations

import pytest

from memforge.memory.destructive_validation import KeptReason, validate_destructive_operations
from memforge.models import RawMemory, ReconcileAction, ReconcileOperation
from memforge.pipeline.revision_assessment import SupportAssessment
from memforge.pipeline.support_relation_coordinator import MemorySupport, memory_support

CLAIM = RawMemory(content="Two reviewers approve payroll.", memory_type="fact")
REPLACEMENT = RawMemory(content="One reviewer approves payroll.", memory_type="fact")

READ_UNSUPPORTED = SupportAssessment(False, "read without Support", None, complete_read=True)
CANDIDATE_ONLY = SupportAssessment(False, "the Candidate's Evidence does not support it", None)
SUPPORTED = SupportAssessment(True, "supported", CLAIM)
PARTIAL = SupportAssessment(None, "unknown part", None, unresolved="partial_coverage")
CAPACITY = SupportAssessment(None, "too large", None, unresolved="capacity")


def _delete() -> ReconcileOperation:
    return ReconcileOperation(action=ReconcileAction.DELETE, memory_id="mem-1", reason="support removed")


def _supersede() -> ReconcileOperation:
    return ReconcileOperation(action=ReconcileAction.SUPERSEDE, memory_id="mem-1", memory=REPLACEMENT)


def _update() -> ReconcileOperation:
    return ReconcileOperation(action=ReconcileAction.UPDATE, memory_id="mem-1", memory=REPLACEMENT)


def _validate(operation, support: MemorySupport | None, *, relation_complete: bool = True):
    supports = {} if support is None else {"mem-1": support}
    return validate_destructive_operations((operation,), supports=supports, relation_complete=relation_complete)


@pytest.mark.parametrize("operation", [_delete(), _supersede()], ids=["delete", "supersede"])
def test_a_removal_resting_on_complete_reads_passes(operation) -> None:
    validation = _validate(operation, memory_support((READ_UNSUPPORTED, READ_UNSUPPORTED)))

    assert validation.operations == (operation,)
    assert validation.kept == {}


def test_a_revision_of_a_supported_claim_passes() -> None:
    validation = _validate(_update(), memory_support((SUPPORTED,)))

    assert validation.operations == (_update(),)


@pytest.mark.parametrize(
    ("operation", "support", "relation_complete", "reason"),
    [
        (_delete(), None, True, KeptReason.SUPPORT_UNRESOLVED),
        (_delete(), MemorySupport(memory_support((READ_UNSUPPORTED,)).result, "no assessments"), True,
         KeptReason.SUPPORT_UNRESOLVED),
        (_delete(), memory_support((READ_UNSUPPORTED, PARTIAL)), True, KeptReason.SUPPORT_UNRESOLVED),
        (_supersede(), memory_support((CAPACITY,)), True, KeptReason.SUPPORT_UNRESOLVED),
        (_update(), memory_support((SUPPORTED, PARTIAL)), True, KeptReason.SUPPORT_UNRESOLVED),
        # Not finding Support in a Candidate's Evidence is no complete read.
        (_delete(), memory_support((READ_UNSUPPORTED, CANDIDATE_ONLY)), True, KeptReason.READ_INCOMPLETE),
        (_supersede(), memory_support((SUPPORTED,)), True, KeptReason.READ_INCOMPLETE),
        (_supersede(), memory_support((READ_UNSUPPORTED,)), False, KeptReason.RELATION_INCOMPLETE),
        (_update(), memory_support((SUPPORTED,)), False, KeptReason.RELATION_INCOMPLETE),
    ],
    ids=[
        "no-support-result", "no-assessments", "partial-coverage", "capacity", "update-partial",
        "candidate-evidence-read", "supported-claim-superseded", "supersede-relation", "update-relation",
    ],
)
def test_an_incomplete_fact_keeps_the_old_memory(operation, support, relation_complete, reason) -> None:
    validation = _validate(operation, support, relation_complete=relation_complete)

    [kept] = validation.operations
    assert kept.action is ReconcileAction.NOOP and kept.memory_id == "mem-1"
    assert kept.memory is None and kept.support_revalidation_skipped
    assert validation.kept == {"mem-1": reason}


def test_removing_support_needs_no_relation_work() -> None:
    validation = _validate(_delete(), memory_support((READ_UNSUPPORTED,)), relation_complete=False)

    assert validation.operations == (_delete(),)


def test_proposals_and_non_destructive_decisions_are_not_validated() -> None:
    review = ReconcileOperation(
        action=ReconcileAction.DELETE, memory_id="mem-1", reason="protected", flag_for_review=True,
    )
    add = ReconcileOperation(action=ReconcileAction.ADD, memory=REPLACEMENT)
    keep = ReconcileOperation(action=ReconcileAction.NOOP, memory_id="mem-2", memory=CLAIM)

    validation = validate_destructive_operations((review, add, keep), supports={}, relation_complete=False)

    assert validation.operations == (review, add, keep)
    assert validation.kept == {}
