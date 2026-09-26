"""SupportRelationCoordinator: every row of the combination table and its precedence."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.memory.evidence import RelationDirection
from memforge.memory.relation_classifier import MemoryRelationType
from memforge.models import CoordinatorProposal, Memory, RawMemory, ReconcileAction, content_hash
from memforge.pipeline.revision_assessment import SupportAssessment
from memforge.pipeline.support_relation_coordinator import (
    MemorySupport,
    RecheckReading,
    RelationLedgerEntry,
    RevisionCompositionProof,
    SupportResult,
    ReconciliationContractError,
    coordinate,
    memory_support,
    plan_rechecks,
    supported_refiners,
)

EQUIVALENT = MemoryRelationType.EQUIVALENT
CONTRADICTS = MemoryRelationType.CONTRADICTS
REFINES = MemoryRelationType.REFINES

SUPPORTED = SupportResult.SUPPORTED
UNAFFECTED = SupportResult.UNAFFECTED
UNSUPPORTED = SupportResult.UNSUPPORTED
CAPACITY = SupportResult.UNRESOLVED_CAPACITY
PARTIAL = SupportResult.UNRESOLVED_PARTIAL_COVERAGE


def _old(memory_id: str = "mem-old") -> Memory:
    content = f"{memory_id} claim"
    now = datetime.now(timezone.utc)
    return Memory(
        id=memory_id, memory_type="fact", content=content, content_hash=content_hash(content),
        created_at=now, updated_at=now,
    )


def _candidate(text: str) -> RawMemory:
    return RawMemory(content=text, memory_type="fact")


# The old claim bound to its verified current Evidence.
REBOUND = RawMemory(content="mem-old claim", memory_type="fact", evidence_anchor="revalidated_noop")


def _support(result: SupportResult, *, rechecked: bool = False) -> MemorySupport:
    evidence = (REBOUND,) if result in {SUPPORTED, UNAFFECTED} else ()
    return MemorySupport(result, f"{result.value} reason", evidence, rechecked)


def _edge(index: int, relation: MemoryRelationType | None, memory_id: str = "mem-old", **kwargs) -> RelationLedgerEntry:
    return RelationLedgerEntry(
        candidate_index=index, incumbent_id=memory_id, relation_type=relation,
        direction=kwargs.pop("direction", RelationDirection.SYMMETRIC), reason=kwargs.pop("reason", ""),
    )


def _coordinate(candidates, relations, supports, *, incumbents=None, proofs=(), rechecked_pairs=frozenset()):
    return coordinate(
        candidates=candidates, incumbents=incumbents or [_old()], relations=relations,
        proofs=proofs, supports=supports, rechecked_pairs=rechecked_pairs,
    )


def _rechecks(candidates, relations, supports, *, incumbents=None, rechecked_pairs=frozenset()):
    return plan_rechecks(
        candidates=candidates, incumbents=incumbents or [_old()], relations=relations,
        supports=supports, rechecked_pairs=rechecked_pairs,
    )


def _by_memory(operations):
    return {operation.memory_id: operation for operation in operations if operation.memory_id}


def _additions(operations):
    return [operation.memory for operation in operations if operation.action is ReconcileAction.ADD]


@pytest.mark.parametrize("result", [SUPPORTED, UNAFFECTED])
def test_a_kept_claim_without_relation_is_rebound_and_the_unrelated_candidate_is_added(result) -> None:
    candidate = _candidate("Retention is seven years.")
    operations = _coordinate([candidate], [], {"mem-old": _support(result)}).operations

    kept = _by_memory(operations)["mem-old"]
    assert kept.action is ReconcileAction.NOOP and kept.memory is REBOUND and not kept.reviews
    assert _additions(operations) == [candidate]


@pytest.mark.parametrize("result", [SUPPORTED, UNAFFECTED])
def test_an_equivalent_of_a_kept_claim_is_consumed(result) -> None:
    operations = _coordinate([_candidate("Two reviewers, restated.")], [_edge(0, EQUIVALENT)], {
        "mem-old": _support(result),
    }).operations

    [kept] = operations
    assert kept.action is ReconcileAction.NOOP and kept.memory is REBOUND and not kept.reviews


def test_a_supported_claim_contradicted_by_the_source_is_rebound_and_reviewed() -> None:
    contradicting = _candidate("One reviewer approves payroll.")
    operations = _coordinate([contradicting], [_edge(0, CONTRADICTS, reason="two versus one")], {
        "mem-old": _support(SUPPORTED),
    }).operations

    [kept] = operations
    assert kept.action is ReconcileAction.NOOP and kept.memory is REBOUND
    [review] = kept.reviews
    assert review.proposal is CoordinatorProposal.SUPERSEDE
    assert review.candidate is contradicting and review.reason == "two versus one"
    assert review.rejection_rebind is None


def test_an_unaffected_claim_contradicted_by_the_source_is_rechecked_in_the_normal_order() -> None:
    relations = [_edge(0, CONTRADICTS)]
    [recheck] = _rechecks([_candidate("One reviewer.")], relations, {"mem-old": _support(UNAFFECTED)})

    assert recheck.memory_id == "mem-old" and recheck.reading is RecheckReading.NORMAL_ORDER
    assert recheck.candidates == ()


@pytest.mark.parametrize(
    ("after_recheck", "expected"),
    [
        (SUPPORTED, "review"),
        (UNSUPPORTED, ReconcileAction.SUPERSEDE),
        (CAPACITY, "kept_unchanged"),
    ],
)
def test_a_normal_order_recheck_result_enters_the_table_again(after_recheck, expected) -> None:
    contradicting = _candidate("One reviewer.")
    relations = [_edge(0, CONTRADICTS)]
    supports = {"mem-old": _support(after_recheck, rechecked=True)}

    assert _rechecks([contradicting], relations, supports) == ()
    operations = _coordinate([contradicting], relations, supports).operations
    kept = _by_memory(operations)["mem-old"]
    if expected == "review":
        assert kept.action is ReconcileAction.NOOP and [r.candidate for r in kept.reviews] == [contradicting]
    elif expected == "kept_unchanged":
        assert kept.support_revalidation_skipped and not kept.reviews
        assert _additions(operations) == []
    else:
        assert kept.action is expected and kept.memory is contradicting


@pytest.mark.parametrize("result", [UNSUPPORTED, PARTIAL])
def test_an_unconfirmed_equivalent_is_rechecked_against_its_candidate_evidence(result) -> None:
    equivalents = [_candidate("Two reviewers, restated."), _candidate("Two reviewers, again.")]
    relations = [_edge(0, EQUIVALENT), _edge(1, EQUIVALENT)]

    [recheck] = _rechecks(equivalents, relations, {"mem-old": _support(result)})

    assert recheck.reading is RecheckReading.CANDIDATE_EVIDENCE
    # One re-check of the claim reads every equivalent Candidate's Evidence.
    assert recheck.candidates == tuple(equivalents)


@pytest.mark.parametrize("result", [UNSUPPORTED, PARTIAL])
def test_a_supported_candidate_evidence_recheck_rebinds_the_claim_and_consumes_the_candidate(result) -> None:
    rechecked = RawMemory(content="mem-old claim", memory_type="fact", evidence_anchor="revalidated_noop")
    support = _support(result).after_candidate_recheck(
        SupportAssessment(True, "Selected current Evidence supports the claim.", rechecked),
    )
    operations = _coordinate([_candidate("Two reviewers, restated.")], [_edge(0, EQUIVALENT)], {
        "mem-old": support,
    }).operations

    [kept] = operations
    assert support.result is SUPPORTED and support.rechecked
    assert kept.action is ReconcileAction.NOOP and kept.memory is rechecked and not kept.reviews


@pytest.mark.parametrize("result", [UNSUPPORTED, PARTIAL])
def test_a_claim_still_unsupported_after_its_recheck_gets_a_rebind_review(result) -> None:
    equivalent = _candidate("Two reviewers, restated.")
    support = _support(result).after_candidate_recheck(
        SupportAssessment(False, "The Candidate's current Evidence does not support the claim.", None),
    )
    relations = [_edge(0, EQUIVALENT)]

    assert support.result is result and support.rechecked
    assert _rechecks([equivalent], relations, {"mem-old": support}) == ()
    [kept] = _coordinate([equivalent], relations, {"mem-old": support}).operations
    assert kept.action is ReconcileAction.NOOP and kept.memory is None and kept.support_revalidation_skipped
    [review] = kept.reviews
    assert review.proposal is CoordinatorProposal.REBIND and review.candidate is equivalent


@pytest.mark.parametrize("result", [UNSUPPORTED, PARTIAL])
def test_a_candidate_evidence_recheck_beyond_capacity_takes_the_capacity_row(result) -> None:
    equivalent, unrelated = _candidate("Two reviewers, restated."), _candidate("Retention is seven years.")
    support = _support(result).after_candidate_recheck(
        SupportAssessment(None, "One ReadingGroup exceeds capacity.", None, unresolved="capacity"),
    )
    relations = [_edge(0, EQUIVALENT)]

    assert support.result is CAPACITY and support.rechecked
    assert _rechecks([equivalent, unrelated], relations, {"mem-old": support}) == ()
    coordination = _coordinate([equivalent, unrelated], relations, {"mem-old": support})
    [kept] = [operation for operation in coordination.operations if operation.memory_id == "mem-old"]
    assert kept.support_revalidation_skipped and not kept.reviews
    assert _additions(coordination.operations) == [unrelated]
    assert coordination.unresolved_candidate_count == 1


def test_an_unsupported_claim_contradicted_by_one_candidate_is_superseded() -> None:
    contradicting = _candidate("One reviewer approves payroll.")
    [operation] = _coordinate([contradicting], [_edge(0, CONTRADICTS)], {
        "mem-old": _support(UNSUPPORTED),
    }).operations

    assert operation.action is ReconcileAction.SUPERSEDE and operation.memory is contradicting
    assert not operation.flag_for_review and not operation.reviews


def test_an_unsupported_claim_without_relation_loses_this_sources_support() -> None:
    [operation] = _coordinate([], [], {"mem-old": _support(UNSUPPORTED)}).operations

    assert operation.action is ReconcileAction.DELETE


def test_a_capacity_claim_is_kept_and_its_related_candidates_are_consumed() -> None:
    equivalent, contradicting, unrelated = (
        _candidate("Two reviewers, restated."), _candidate("One reviewer."), _candidate("Retention is seven years."),
    )
    relations = [_edge(0, EQUIVALENT), _edge(1, CONTRADICTS)]
    supports = {"mem-old": _support(CAPACITY)}

    assert _rechecks([equivalent, contradicting, unrelated], relations, supports) == ()
    coordination = _coordinate([equivalent, contradicting, unrelated], relations, supports)
    kept = _by_memory(coordination.operations)["mem-old"]
    assert kept.support_revalidation_skipped and not kept.reviews and kept.reason == "unresolved_capacity reason"
    assert _additions(coordination.operations) == [unrelated]
    assert coordination.unresolved_candidate_count == 2


def test_a_partial_coverage_claim_contradicted_by_the_source_gets_a_supersession_review() -> None:
    contradicting = _candidate("One reviewer.")
    relations = [_edge(0, CONTRADICTS)]
    supports = {"mem-old": _support(PARTIAL)}

    assert _rechecks([contradicting], relations, supports) == ()
    [kept] = _coordinate([contradicting], relations, supports).operations
    assert kept.action is ReconcileAction.NOOP and kept.support_revalidation_skipped
    [review] = kept.reviews
    assert review.proposal is CoordinatorProposal.SUPERSEDE and review.candidate is contradicting


def test_a_partial_coverage_claim_without_relation_is_kept_unchanged() -> None:
    [kept] = _coordinate([], [], {"mem-old": _support(PARTIAL)}).operations

    assert kept.action is ReconcileAction.NOOP and kept.support_revalidation_skipped and not kept.reviews


@pytest.mark.parametrize("result", [SUPPORTED, UNAFFECTED, UNSUPPORTED, PARTIAL])
def test_an_old_memory_with_both_edges_is_reviewed_before_any_other_row(result) -> None:
    equivalent, contradicting = _candidate("Two reviewers, restated."), _candidate("One reviewer.")
    relations = [_edge(0, EQUIVALENT), _edge(1, CONTRADICTS)]
    supports = {"mem-old": _support(result)}

    assert _rechecks([equivalent, contradicting], relations, supports) == ()
    [kept] = _coordinate([equivalent, contradicting], relations, supports).operations
    assert kept.action is ReconcileAction.NOOP
    assert (kept.memory is REBOUND) is (result in {SUPPORTED, UNAFFECTED})
    [review] = kept.reviews
    assert review.proposal is CoordinatorProposal.SUPERSEDE and review.candidate is contradicting
    assert review.rejection_rebind is equivalent


def test_capacity_takes_precedence_over_both_edges() -> None:
    relations = [_edge(0, EQUIVALENT), _edge(1, CONTRADICTS)]
    [kept] = _coordinate([_candidate("a"), _candidate("b")], relations, {"mem-old": _support(CAPACITY)}).operations

    assert kept.support_revalidation_skipped and not kept.reviews


def test_a_candidate_treated_differently_by_two_old_memories_leaves_the_component_unresolved() -> None:
    candidate, unrelated = _candidate("Two reviewers approve payroll."), _candidate("Retention is seven years.")
    first, second, third = _old("mem-a"), _old("mem-b"), _old("mem-c")
    # Consumed as an equivalent by mem-a, but the replacement of an unsupported mem-b.
    relations = [_edge(0, EQUIVALENT, "mem-a"), _edge(0, CONTRADICTS, "mem-b")]
    coordination = _coordinate([candidate, unrelated], relations, {
        "mem-a": _support(SUPPORTED), "mem-b": _support(UNSUPPORTED), "mem-c": _support(UNSUPPORTED),
    }, incumbents=[first, second, third])

    by_memory = _by_memory(coordination.operations)
    for memory_id in ("mem-a", "mem-b"):
        assert by_memory[memory_id].action is ReconcileAction.NOOP
        assert by_memory[memory_id].support_revalidation_skipped and not by_memory[memory_id].reviews
    assert by_memory["mem-c"].action is ReconcileAction.DELETE
    assert _additions(coordination.operations) == [unrelated]
    assert coordination.unresolved_candidate_count == 1


@pytest.mark.parametrize(
    ("second_relation", "second_support"),
    [(CONTRADICTS, _support(SUPPORTED)), (EQUIVALENT, _support(UNSUPPORTED, rechecked=True))],
    ids=["two-supersessions", "supersession-and-rebind"],
)
def test_a_candidate_staged_in_the_reviews_of_two_old_memories_leaves_the_component_unresolved(
    second_relation, second_support,
) -> None:
    # Each approval would apply the same Candidate again: create it twice, or bind a second claim to it.
    staged = _candidate("One reviewer approves payroll.")
    relations = [_edge(0, CONTRADICTS, "mem-a"), _edge(0, second_relation, "mem-b")]
    coordination = _coordinate([staged], relations, {
        "mem-a": _support(SUPPORTED), "mem-b": second_support,
    }, incumbents=[_old("mem-a"), _old("mem-b")])

    for operation in coordination.operations:
        assert operation.action is ReconcileAction.NOOP
        assert operation.support_revalidation_skipped and not operation.reviews
    assert _additions(coordination.operations) == []
    assert coordination.unresolved_candidate_count == 1


def test_refinements_of_a_claim_in_an_unresolved_component_are_not_compared() -> None:
    first, second = _candidate("Two reviewers from team A."), _candidate("Two reviewers from team B.")
    refines = {"direction": RelationDirection.CHALLENGER_TO_CANDIDATE}
    relations = [_edge(0, REFINES, **refines), _edge(1, REFINES, **refines)]
    supports = {"mem-old": _support(SUPPORTED)}
    ledger = {"candidates": [first, second], "incumbents": [_old()], "supports": supports}

    assert supported_refiners(relations=relations, **ledger) == {"mem-old": (0, 1)}
    # An uncertain edge of one refiner leaves the old Memory undecided this round.
    unresolved = [*relations[:1], _edge(1, None)]
    assert supported_refiners(relations=unresolved, **ledger) == {}


def test_one_candidate_consumed_by_two_kept_claims_is_one_treatment() -> None:
    relations = [_edge(0, EQUIVALENT, "mem-a"), _edge(0, EQUIVALENT, "mem-b")]
    coordination = _coordinate([_candidate("Retries back off.")], relations, {
        "mem-a": _support(SUPPORTED), "mem-b": _support(SUPPORTED),
    }, incumbents=[_old("mem-a"), _old("mem-b")])

    assert all(not operation.support_revalidation_skipped for operation in coordination.operations)
    assert coordination.unresolved_candidate_count == 0


def test_a_lossless_refinement_of_a_kept_claim_revises_it() -> None:
    refinement = _candidate("Two reviewers from distinct teams approve payroll.")
    relations = [_edge(0, REFINES, direction=RelationDirection.CHALLENGER_TO_CANDIDATE)]
    proof = RevisionCompositionProof(0, "mem-old", True, True, True, reason="adds the team condition")

    [operation] = _coordinate([refinement], relations, {"mem-old": _support(SUPPORTED)}, proofs=[proof]).operations

    assert operation.action is ReconcileAction.UPDATE and operation.memory is refinement


def test_a_refinement_preserving_an_unsupported_claim_stays_unresolved() -> None:
    refinement = _candidate("Two reviewers from distinct teams approve payroll.")
    relations = [_edge(0, REFINES, direction=RelationDirection.CHALLENGER_TO_CANDIDATE)]
    proof = RevisionCompositionProof(0, "mem-old", False, True, False)

    coordination = _coordinate([refinement], relations, {"mem-old": _support(UNSUPPORTED)}, proofs=[proof])

    [kept] = coordination.operations
    assert kept.support_revalidation_skipped
    assert coordination.unresolved_candidate_count == 1


def test_an_uncertain_relation_keeps_its_component_and_never_adds() -> None:
    [kept] = _coordinate([_candidate("Maybe related.")], [_edge(0, None)], {"mem-old": _support(UNSUPPORTED)}).operations

    assert kept.action is ReconcileAction.NOOP and kept.support_revalidation_skipped


def test_a_carried_conflict_is_not_rechecked_again() -> None:
    relations = [_edge(0, EQUIVALENT)]
    assert _rechecks([_candidate("x")], relations, {"mem-old": _support(UNSUPPORTED)},
                     rechecked_pairs=frozenset({(0, "mem-old")})) == ()
    relations = [_edge(0, CONTRADICTS)]
    assert _rechecks([_candidate("x")], relations, {"mem-old": _support(UNAFFECTED)},
                     rechecked_pairs=frozenset({(0, "mem-old")})) == ()


def _assessment(supported, *, unresolved=None, rebound=False):
    memory = REBOUND if supported else None
    return SupportAssessment(supported, "reason", memory, unresolved=unresolved, rebound=rebound)


@pytest.mark.parametrize(
    ("assessments", "expected"),
    [
        ([_assessment(True), _assessment(None, unresolved="capacity")], CAPACITY),
        ([_assessment(True), _assessment(None, unresolved="partial_coverage")], PARTIAL),
        ([_assessment(None, unresolved="partial_coverage"), _assessment(None, unresolved="capacity")], CAPACITY),
        ([_assessment(True, rebound=True), _assessment(True)], SUPPORTED),
        ([_assessment(True, rebound=True), _assessment(True, rebound=True)], UNAFFECTED),
        ([_assessment(True, rebound=True), _assessment(False)], UNAFFECTED),
        ([_assessment(False), _assessment(False)], UNSUPPORTED),
    ],
)
def test_several_supports_of_one_claim_combine_by_precedence(assessments, expected) -> None:
    assert memory_support(assessments).result is expected


def test_an_incomplete_support_ledger_fails_closed() -> None:
    with pytest.raises(ReconciliationContractError):
        _coordinate([], [], {})
