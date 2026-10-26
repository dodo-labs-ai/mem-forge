"""SupportRelationCoordinator: every row of the combination table and its precedence."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.memory.evidence import RelationDirection
from memforge.memory.relation_classifier import SameUnitRelationType
from memforge.models import Memory, RawMemory, ReconcileAction, content_hash
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
    revisions_to_verify,
    supported_refiners,
)

EQUIVALENT = SameUnitRelationType.EQUIVALENT
CONTRADICTS = SameUnitRelationType.CONTRADICTS
REFINES = SameUnitRelationType.REFINES

SUPPORTED = SupportResult.SUPPORTED
UNAFFECTED = SupportResult.UNAFFECTED
UNSUPPORTED = SupportResult.UNSUPPORTED
CAPACITY = SupportResult.UNRESOLVED_CAPACITY
INVALID = SupportResult.UNRESOLVED_INVALID_RESPONSE
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


def _edge(index: int, relation: SameUnitRelationType | None, memory_id: str = "mem-old", **kwargs) -> RelationLedgerEntry:
    return RelationLedgerEntry(
        candidate_index=index, incumbent_id=memory_id, relation_type=relation,
        direction=kwargs.pop("direction", RelationDirection.SYMMETRIC), reason=kwargs.pop("reason", ""),
    )


def _coordinate(candidates, relations, supports, *, incumbents=None, proofs=(), unjudged_candidates=frozenset()):
    return coordinate(
        candidates=candidates, incumbents=incumbents or [_old()], relations=relations,
        proofs=proofs, supports=supports, unjudged_candidates=unjudged_candidates,
    )


def _rechecks(candidates, relations, supports, *, incumbents=None):
    return plan_rechecks(
        candidates=candidates, incumbents=incumbents or [_old()], relations=relations, supports=supports,
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
    assert kept.action is ReconcileAction.NOOP and kept.memory is REBOUND
    assert _additions(operations) == [candidate]


@pytest.mark.parametrize("result", [SUPPORTED, UNAFFECTED])
def test_an_equivalent_of_a_kept_claim_is_consumed(result) -> None:
    operations = _coordinate([_candidate("Two reviewers, restated.")], [_edge(0, EQUIVALENT)], {
        "mem-old": _support(result),
    }).operations

    [kept] = operations
    assert kept.action is ReconcileAction.NOOP and kept.memory is REBOUND


def test_a_supported_claim_contradicted_by_the_source_is_kept_and_the_candidate_added() -> None:
    # The current source states both claims; relation discovery relates the two Memories after commit.
    contradicting = _candidate("One reviewer approves payroll.")
    operations = _coordinate([contradicting], [_edge(0, CONTRADICTS, reason="two versus one")], {
        "mem-old": _support(SUPPORTED),
    }).operations

    kept = _by_memory(operations)["mem-old"]
    assert kept.action is ReconcileAction.NOOP and kept.memory is REBOUND
    assert _additions(operations) == [contradicting]


def test_an_unaffected_claim_contradicted_by_the_source_is_rechecked_in_the_normal_order() -> None:
    relations = [_edge(0, CONTRADICTS)]
    [recheck] = _rechecks([_candidate("One reviewer.")], relations, {"mem-old": _support(UNAFFECTED)})

    assert recheck.memory_id == "mem-old" and recheck.reading is RecheckReading.NORMAL_ORDER
    assert recheck.candidates == ()


@pytest.mark.parametrize(
    ("after_recheck", "expected"),
    [
        (SUPPORTED, "kept_and_added"),
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
    if expected == "kept_and_added":
        assert kept.action is ReconcileAction.NOOP and kept.memory is REBOUND
        assert _additions(operations) == [contradicting]
    elif expected == "kept_unchanged":
        assert kept.support_revalidation_skipped
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
    assert kept.action is ReconcileAction.NOOP and kept.memory is rechecked


def _still_unsupported(result: SupportResult) -> MemorySupport:
    return _support(result).after_candidate_recheck(
        SupportAssessment(False, "The Candidate's current Evidence does not support the claim.", None),
    )


def test_an_unsupported_claim_whose_recheck_finds_nothing_loses_support_and_its_equivalent_is_added() -> None:
    equivalent = _candidate("Two reviewers, restated.")
    support = _still_unsupported(UNSUPPORTED)
    relations = [_edge(0, EQUIVALENT)]

    assert support.result is UNSUPPORTED and support.rechecked
    assert _rechecks([equivalent], relations, {"mem-old": support}) == ()
    operations = _coordinate([equivalent], relations, {"mem-old": support}).operations
    assert _by_memory(operations)["mem-old"].action is ReconcileAction.DELETE
    assert _additions(operations) == [equivalent]


def test_a_partial_coverage_claim_whose_recheck_finds_nothing_is_kept_and_its_equivalent_consumed() -> None:
    equivalent = _candidate("Two reviewers, restated.")
    support = _still_unsupported(PARTIAL)
    relations = [_edge(0, EQUIVALENT)]

    assert support.result is PARTIAL and support.rechecked
    assert _rechecks([equivalent], relations, {"mem-old": support}) == ()
    [kept] = _coordinate([equivalent], relations, {"mem-old": support}).operations
    assert kept.action is ReconcileAction.NOOP and kept.memory is None and kept.support_revalidation_skipped


@pytest.mark.parametrize("result", [UNSUPPORTED, PARTIAL])
@pytest.mark.parametrize("unjudged", ["capacity", "invalid_response"])
def test_a_candidate_evidence_recheck_that_cannot_be_judged_takes_the_unjudged_row(result, unjudged) -> None:
    equivalent, unrelated = _candidate("Two reviewers, restated."), _candidate("Retention is seven years.")
    support = _support(result).after_candidate_recheck(
        SupportAssessment(None, "One ReadingGroup cannot be judged.", None, unresolved=unjudged),
    )
    relations = [_edge(0, EQUIVALENT)]

    assert support.result is (CAPACITY if unjudged == "capacity" else INVALID) and support.rechecked
    assert _rechecks([equivalent, unrelated], relations, {"mem-old": support}) == ()
    coordination = _coordinate([equivalent, unrelated], relations, {"mem-old": support})
    [kept] = [operation for operation in coordination.operations if operation.memory_id == "mem-old"]
    assert kept.support_revalidation_skipped
    assert _additions(coordination.operations) == [unrelated]
    assert coordination.unresolved_candidate_count == 1


def test_an_unsupported_claim_contradicted_by_one_candidate_is_superseded() -> None:
    contradicting = _candidate("One reviewer approves payroll.")
    [operation] = _coordinate([contradicting], [_edge(0, CONTRADICTS)], {
        "mem-old": _support(UNSUPPORTED),
    }).operations

    assert operation.action is ReconcileAction.SUPERSEDE and operation.memory is contradicting


def test_an_unsupported_claim_without_relation_loses_this_sources_support() -> None:
    [operation] = _coordinate([], [], {"mem-old": _support(UNSUPPORTED)}).operations

    assert operation.action is ReconcileAction.DELETE


@pytest.mark.parametrize("result", [CAPACITY, INVALID])
def test_an_unjudged_claim_is_kept_and_its_related_candidates_are_consumed(result) -> None:
    equivalent, contradicting, unrelated = (
        _candidate("Two reviewers, restated."), _candidate("One reviewer."), _candidate("Retention is seven years."),
    )
    relations = [_edge(0, EQUIVALENT), _edge(1, CONTRADICTS)]
    supports = {"mem-old": _support(result)}

    assert _rechecks([equivalent, contradicting, unrelated], relations, supports) == ()
    coordination = _coordinate([equivalent, contradicting, unrelated], relations, supports)
    kept = _by_memory(coordination.operations)["mem-old"]
    assert kept.support_revalidation_skipped and kept.reason == f"{result.value} reason"
    assert _additions(coordination.operations) == [unrelated]
    assert coordination.unresolved_candidate_count == 2


def test_an_unjudged_candidate_withholds_every_destructive_action_of_the_unit() -> None:
    from memforge.memory.destructive_validation import KeptReason, validate_destructive_operations

    unjudged, contradicting, unrelated = (
        _candidate("mem-gone claim, restated."), _candidate("One reviewer."), _candidate("Retention is seven years."),
    )
    incumbents = [_old("mem-old"), _old("mem-gone")]
    read_unsupported = memory_support((SupportAssessment(False, "read whole order", None, complete_read=True),))
    supports = {"mem-old": read_unsupported, "mem-gone": read_unsupported}

    coordination = _coordinate(
        [unjudged, contradicting, unrelated], [_edge(1, CONTRADICTS)], supports,
        incumbents=incumbents, unjudged_candidates=frozenset({0}),
    )
    # The table still decides every old Memory, and the unjudged Candidate is consumed.
    decided = _by_memory(coordination.operations)
    assert (decided["mem-old"].action, decided["mem-gone"].action) == (ReconcileAction.SUPERSEDE, ReconcileAction.DELETE)
    assert coordination.unresolved_candidate_count == 1

    # The Relation line lacks the unjudged Candidate's row, so no destructive decision is executed:
    # the unjudged Candidate may restate mem-gone, which must not disappear with it.
    validation = validate_destructive_operations(coordination.operations, supports=supports, relation_complete=False)
    kept = _by_memory(validation.operations)
    assert all(kept[memory_id].action is ReconcileAction.NOOP and kept[memory_id].support_revalidation_skipped
               for memory_id in ("mem-old", "mem-gone"))
    assert validation.kept == dict.fromkeys(("mem-old", "mem-gone"), KeptReason.RELATION_INCOMPLETE)
    # The withheld SUPERSEDE was the contradicting Candidate's only effect, so it is not added either.
    assert _additions(validation.operations) == [unrelated]


def test_an_unjudged_candidate_outside_the_ledger_fails_closed() -> None:
    with pytest.raises(ReconciliationContractError):
        _coordinate([], [], {"mem-old": _support(SUPPORTED)}, unjudged_candidates=frozenset({0}))


def test_a_partial_coverage_claim_contradicted_by_the_source_is_kept_and_the_candidate_added() -> None:
    contradicting = _candidate("One reviewer.")
    relations = [_edge(0, CONTRADICTS)]
    supports = {"mem-old": _support(PARTIAL)}

    assert _rechecks([contradicting], relations, supports) == ()
    operations = _coordinate([contradicting], relations, supports).operations
    kept = _by_memory(operations)["mem-old"]
    assert kept.action is ReconcileAction.NOOP and kept.support_revalidation_skipped
    assert _additions(operations) == [contradicting]


def test_a_partial_coverage_claim_without_relation_is_kept_unchanged() -> None:
    [kept] = _coordinate([], [], {"mem-old": _support(PARTIAL)}).operations

    assert kept.action is ReconcileAction.NOOP and kept.support_revalidation_skipped


@pytest.mark.parametrize(
    ("support", "action", "added"),
    [
        (_support(SUPPORTED), ReconcileAction.NOOP, "contradicting"),
        (_still_unsupported(PARTIAL), ReconcileAction.NOOP, "contradicting"),
        (_still_unsupported(UNSUPPORTED), ReconcileAction.DELETE, "both"),
    ],
    ids=["supported", "partial_coverage", "unsupported"],
)
def test_an_old_memory_with_both_edges_takes_its_own_support_row(support, action, added) -> None:
    equivalent, contradicting = _candidate("Two reviewers, restated."), _candidate("One reviewer.")
    relations = [_edge(0, EQUIVALENT), _edge(1, CONTRADICTS)]
    supports = {"mem-old": support}

    assert _rechecks([equivalent, contradicting], relations, supports) == ()
    operations = _coordinate([equivalent, contradicting], relations, supports).operations
    assert _by_memory(operations)["mem-old"].action is action
    # The equivalent restates a kept claim; two Candidates name no single successor of an unsupported one.
    assert _additions(operations) == ([contradicting] if added == "contradicting" else [equivalent, contradicting])


def test_capacity_takes_precedence_over_both_edges() -> None:
    relations = [_edge(0, EQUIVALENT), _edge(1, CONTRADICTS)]
    [kept] = _coordinate([_candidate("a"), _candidate("b")], relations, {"mem-old": _support(CAPACITY)}).operations

    assert kept.support_revalidation_skipped


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
    # The supported claim keeps its own Support result; the unsupported one stays unchanged.
    assert by_memory["mem-a"].memory is REBOUND and not by_memory["mem-a"].support_revalidation_skipped
    assert by_memory["mem-b"].memory is None and by_memory["mem-b"].support_revalidation_skipped
    assert by_memory["mem-c"].action is ReconcileAction.DELETE
    assert _additions(coordination.operations) == [unrelated]
    assert coordination.unresolved_candidate_count == 1


def test_a_candidate_contradicting_two_kept_claims_is_added_once() -> None:
    contradicting = _candidate("One reviewer approves payroll.")
    relations = [_edge(0, CONTRADICTS, "mem-a"), _edge(0, CONTRADICTS, "mem-b")]
    coordination = _coordinate([contradicting], relations, {
        "mem-a": _support(SUPPORTED), "mem-b": _support(SUPPORTED),
    }, incumbents=[_old("mem-a"), _old("mem-b")])

    by_memory = _by_memory(coordination.operations)
    assert all(by_memory[memory_id].memory is REBOUND for memory_id in ("mem-a", "mem-b"))
    assert _additions(coordination.operations) == [contradicting]
    assert coordination.unresolved_candidate_count == 0


def test_a_candidate_merging_several_kept_claims_leaves_each_on_its_verified_evidence() -> None:
    # One Candidate restates three old claims and losslessly refines a fourth.
    merged = _candidate("Two reviewers from distinct teams approve payroll and retention is seven years.")
    memory_ids = ("mem-a", "mem-b", "mem-c", "mem-d")
    relations = [
        *(_edge(0, EQUIVALENT, memory_id) for memory_id in memory_ids[:3]),
        _edge(0, REFINES, "mem-d", direction=RelationDirection.CHALLENGER_TO_CANDIDATE),
    ]
    proof = RevisionCompositionProof(0, "mem-d", True, True, True)
    coordination = _coordinate(
        [merged], relations, {memory_id: _support(SUPPORTED) for memory_id in memory_ids},
        incumbents=[_old(memory_id) for memory_id in memory_ids], proofs=[proof],
    )

    for operation in coordination.operations:
        assert operation.action is ReconcileAction.NOOP and operation.memory is REBOUND
        assert not operation.support_revalidation_skipped
    assert _additions(coordination.operations) == []
    assert coordination.unresolved_candidate_count == 1


@pytest.mark.parametrize("result", [SUPPORTED, UNAFFECTED])
def test_a_kept_claim_without_verified_evidence_stays_unchanged_in_an_unresolved_component(result) -> None:
    support = MemorySupport(result, f"{result.value} reason")

    [kept] = _coordinate([_candidate("Maybe related.")], [_edge(0, None)], {"mem-old": support}).operations

    assert kept.memory is None and kept.support_revalidation_skipped


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


@pytest.mark.parametrize("result", [SUPPORTED, UNAFFECTED, UNSUPPORTED, PARTIAL])
def test_an_uncertain_relation_keeps_its_component_and_never_adds(result) -> None:
    coordination = _coordinate([_candidate("Maybe related.")], [_edge(0, None)], {"mem-old": _support(result)})

    [kept] = coordination.operations
    assert kept.action is ReconcileAction.NOOP
    # Only a Support result that keeps the claim moves it to current Evidence.
    assert (kept.memory is REBOUND) is (result in {SUPPORTED, UNAFFECTED})
    assert kept.support_revalidation_skipped is (result in {UNSUPPORTED, PARTIAL})
    assert coordination.unresolved_candidate_count == 1


def test_an_unaffected_claim_with_both_edges_is_rechecked_in_the_normal_order() -> None:
    relations = [_edge(0, EQUIVALENT), _edge(1, CONTRADICTS)]
    [recheck] = _rechecks([_candidate("a"), _candidate("b")], relations, {"mem-old": _support(UNAFFECTED)})

    assert recheck.reading is RecheckReading.NORMAL_ORDER


@pytest.mark.parametrize(
    ("relation", "result"), [(EQUIVALENT, UNSUPPORTED), (CONTRADICTS, UNAFFECTED)],
)
def test_a_rechecked_claim_is_not_rechecked_again(relation, result) -> None:
    assert _rechecks([_candidate("x")], [_edge(0, relation)], {"mem-old": _support(result, rechecked=True)}) == ()


def _assessment(supported, *, unresolved=None, rebound=False):
    memory = REBOUND if supported else None
    return SupportAssessment(supported, "reason", memory, unresolved=unresolved, rebound=rebound)


@pytest.mark.parametrize(
    ("assessments", "expected"),
    [
        ([_assessment(True), _assessment(None, unresolved="capacity")], CAPACITY),
        ([_assessment(True), _assessment(None, unresolved="partial_coverage")], PARTIAL),
        ([_assessment(None, unresolved="partial_coverage"), _assessment(None, unresolved="capacity")], CAPACITY),
        ([_assessment(True), _assessment(None, unresolved="invalid_response")], INVALID),
        ([_assessment(None, unresolved="partial_coverage"), _assessment(None, unresolved="invalid_response")], INVALID),
        ([_assessment(None, unresolved="invalid_response"), _assessment(None, unresolved="capacity")], CAPACITY),
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


def test_every_revision_the_table_would_apply_is_verified_and_a_dropped_one_is_not_applied() -> None:
    refines = dict(direction=RelationDirection.CHALLENGER_TO_CANDIDATE)
    candidates = [_candidate("c refines m1"), _candidate("d refines m2")]
    incumbents = [_old("m1"), _old("m2"), _old("m3")]
    # c also restates m2, so a divided treatment would hold c back; d's revision of m2 is still verified.
    relations = [
        _edge(0, REFINES, "m1", **refines), _edge(0, EQUIVALENT, "m2"), _edge(1, REFINES, "m2", **refines),
        _edge(1, CONTRADICTS, "m3"),
    ]
    proofs = [RevisionCompositionProof(0, "m1", True, True, True), RevisionCompositionProof(1, "m2", True, True, True)]
    supports = {"m1": _support(SUPPORTED), "m2": _support(UNAFFECTED), "m3": _support(SUPPORTED)}

    pairs = revisions_to_verify(
        candidates=candidates, incumbents=incumbents, relations=relations, proofs=proofs, supports=supports,
    )
    assert pairs == ((0, "m1"), (1, "m2"))

    single = dict(candidates=[candidates[0]], incumbents=[_old("m1")], relations=[relations[0]],
                  proofs=[proofs[0]], supports={"m1": _support(SUPPORTED)})
    verified = coordinate(**single)
    dropped = coordinate(**single, unverified_revisions=frozenset({(0, "m1")}))

    assert _by_memory(verified.operations)["m1"].action is ReconcileAction.UPDATE
    assert _by_memory(dropped.operations)["m1"].action is ReconcileAction.NOOP
    assert _additions(dropped.operations) == [candidates[0]]


@pytest.mark.parametrize(
    ("result", "relations", "proof"),
    [
        (UNSUPPORTED, [REFINES], RevisionCompositionProof(0, "mem-old", True, True, True)),
        (PARTIAL, [REFINES], RevisionCompositionProof(0, "mem-old", True, True, True)),
        (SUPPORTED, [REFINES], RevisionCompositionProof(0, "mem-old", True, False, True)),
        (SUPPORTED, [REFINES, CONTRADICTS], RevisionCompositionProof(0, "mem-old", True, True, True)),
    ],
)
def test_no_pair_is_verified_unless_the_table_would_revise_it(result, relations, proof) -> None:
    refines = dict(direction=RelationDirection.CHALLENGER_TO_CANDIDATE)
    edges = [_edge(index, relation, **(refines if relation is REFINES else {}))
             for index, relation in enumerate(relations)]

    assert revisions_to_verify(
        candidates=[_candidate(f"c{index}") for index in range(len(relations))], incumbents=[_old()],
        relations=edges, proofs=[proof], supports={"mem-old": _support(result)},
    ) == ()


def test_a_contradicted_claim_is_never_revised_by_its_refinement() -> None:
    refinement, contradicting = _candidate("Two reviewers from distinct teams."), _candidate("One reviewer.")
    relations = [_edge(0, REFINES, direction=RelationDirection.CHALLENGER_TO_CANDIDATE), _edge(1, CONTRADICTS)]
    proof = RevisionCompositionProof(0, "mem-old", True, True, True)

    operations = _coordinate([refinement, contradicting], relations, {"mem-old": _support(SUPPORTED)},
                             proofs=[proof]).operations

    kept = _by_memory(operations)["mem-old"]
    assert kept.action is ReconcileAction.NOOP and kept.memory is REBOUND
    assert _additions(operations) == [refinement, contradicting]


def test_refinements_of_a_contradicted_claim_are_not_compared() -> None:
    refines = {"direction": RelationDirection.CHALLENGER_TO_CANDIDATE}
    relations = [_edge(0, REFINES, **refines), _edge(1, REFINES, **refines), _edge(2, CONTRADICTS)]

    assert supported_refiners(
        candidates=[_candidate("a"), _candidate("b"), _candidate("c")], incumbents=[_old()],
        relations=relations, supports={"mem-old": _support(SUPPORTED)},
    ) == {}
