"""Join the Support line and the Relation line of one Source Unit revision.

SupportRelationCoordinator is program code. It runs after both lines finish and
combines, for every same-Unit old Memory, its Memory-level Support result with
the Relation edges that point at it. Neither line decides the other's truth.

Rows are matched in precedence order and the first match wins:

1. ``UNRESOLVED(capacity)``: keep unchanged; related Candidates are consumed.
2. An equivalent and a contradicts edge on the same old Memory: coordinator
   Review that stages the contradicting Candidate and proposes supersession;
   rejection rebinds the old Memory to the equivalent Candidate's Evidence.
3. The remaining rows:

   ======================================  ============  ==========================================
   Support                                 Relation      Action
   ======================================  ============  ==========================================
   SUPPORTED, or UNAFFECTED                none, equiv.  keep and rebind; equivalents consumed
   SUPPORTED                               contradicts   keep and rebind; Review proposes supersession
   UNAFFECTED                              contradicts   re-check in the normal reading order
   UNSUPPORTED                             equivalent    re-check with the Candidate's Evidence
   UNSUPPORTED                             contradicts   SUPERSEDE
   UNSUPPORTED                             none          remove this source's Support
   UNRESOLVED(partial_coverage)            equivalent    re-check with the Candidate's Evidence
   UNRESOLVED(partial_coverage)            contradicts   keep; Review proposes supersession
   UNRESOLVED(partial_coverage)            none          keep unchanged
   ======================================  ============  ==========================================

A claim is re-checked at most once per revision. A supported re-check with a
Candidate's Evidence keeps the old Memory, binds it to that Evidence and
consumes the Candidate; a claim still unsupported gets a Review that proposes
exactly that rebind. A normal-order re-check yields an ordinary Support result
that enters the table again. REFINES and uncertain edges keep the local
unresolved relationship rules. A Candidate that receives different treatments
across old Memories, or is staged in the Reviews of more than one, leaves its
whole related component unresolved: consumed this round, with no ADD and no
destructive action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType

from memforge.memory.evidence import RelationDirection
from memforge.memory.relation_classifier import MemoryRelationType
from memforge.models import (
    CoordinatorProposal,
    CoordinatorReview,
    Memory,
    RawMemory,
    ReconcileAction,
    ReconcileOperation,
)
from memforge.pipeline.revision_assessment import SupportAssessment

__all__ = [
    "UNRESOLVED_RESULTS",
    "Coordination",
    "MemorySupport",
    "RecheckReading",
    "ReconciliationContractError",
    "RelationLedgerEntry",
    "RevisionCompositionProof",
    "SupportRecheckRequest",
    "SupportResult",
    "coordinate",
    "memory_support",
    "plan_rechecks",
    "supported_refiners",
]


class ReconciliationContractError(ValueError):
    """A bounded fail-closed reconciliation invariant violation."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class SupportResult(str, Enum):
    """One old Memory's Support result in this revision."""

    # A read found current Evidence that completely supports the claim.
    SUPPORTED = "supported"
    # Every read Support was bound to exactly unchanged Evidence without a read.
    UNAFFECTED = "unaffected"
    # The whole reading order was read without complete Support.
    UNSUPPORTED = "unsupported"
    # One ReadingGroup alone exceeds the model's capacity for the claim.
    UNRESOLVED_CAPACITY = "unresolved_capacity"
    # A prior Evidence part is UNKNOWN under partial projection coverage.
    UNRESOLVED_PARTIAL_COVERAGE = "unresolved_partial_coverage"

    @property
    def unresolved(self) -> bool:
        return self in UNRESOLVED_RESULTS.values()


# A Support Assessment's unresolved reason and the Memory-level result it yields, in precedence order.
UNRESOLVED_RESULTS: Mapping[str, SupportResult] = MappingProxyType({
    "capacity": SupportResult.UNRESOLVED_CAPACITY,
    "partial_coverage": SupportResult.UNRESOLVED_PARTIAL_COVERAGE,
})
_KEEPS_SUPPORT = frozenset({SupportResult.SUPPORTED, SupportResult.UNAFFECTED})


@dataclass(frozen=True)
class MemorySupport:
    """The Memory-level result of every Support one old Memory has in this Source Unit."""

    result: SupportResult
    reason: str
    # The claim bound to the current Evidence of each Support that keeps it; the first leads.
    evidence: tuple[RawMemory, ...] = ()
    # This revision already re-checked the claim; a remaining conflict goes to Review.
    rechecked: bool = False
    # The Support assessments this result rests on; DestructiveValidation verifies them.
    assessments: tuple[SupportAssessment, ...] = ()

    def after_candidate_recheck(self, assessment: SupportAssessment) -> MemorySupport:
        """The result once the claim was read against its equivalent Candidates' current Evidence.

        Found Support replaces the old Evidence, including a part that was UNKNOWN.
        Not finding it there says nothing about the rest of the revision.
        """
        if assessment.supported and assessment.memory is not None:
            return MemorySupport(
                result=SupportResult.SUPPORTED, reason=assessment.reason, evidence=(assessment.memory,),
                rechecked=True, assessments=(assessment,),
            )
        if assessment.unresolved == "capacity":
            return MemorySupport(
                result=SupportResult.UNRESOLVED_CAPACITY, reason=assessment.reason, rechecked=True,
                assessments=(assessment,),
            )
        return replace(self, rechecked=True)


def memory_support(assessments: Sequence[SupportAssessment], *, rechecked: bool = False) -> MemorySupport:
    """Combine one old Memory's Support assessments by the table's precedence.

    Any ``UNRESOLVED(capacity)`` Support decides the Memory, then any
    ``UNRESOLVED(partial_coverage)`` one. Otherwise a read that found Support
    makes it SUPPORTED; Supports that were only rebound make it UNAFFECTED, so a
    contradiction re-reads those. Only when every Support was read without
    complete Support is the Memory UNSUPPORTED.
    """
    reason = "; ".join(dict.fromkeys(assessment.reason for assessment in assessments))
    assessments = tuple(assessments)
    evidence = tuple(assessment.memory for assessment in assessments if assessment.supported and assessment.memory)
    result = next(
        (
            unresolved_result for unresolved, unresolved_result in UNRESOLVED_RESULTS.items()
            if any(assessment.unresolved == unresolved for assessment in assessments)
        ),
        None,
    )
    if result is not None:
        evidence = ()
    elif any(assessment.supported and not assessment.rebound for assessment in assessments):
        result = SupportResult.SUPPORTED
    elif evidence:
        result = SupportResult.UNAFFECTED
    else:
        result = SupportResult.UNSUPPORTED
    return MemorySupport(
        result=result, reason=reason, evidence=evidence, rechecked=rechecked, assessments=assessments,
    )


@dataclass(frozen=True, slots=True)
class RelationLedgerEntry:
    """One complete, datastore-bound candidate/incumbent relation; ``None`` is explicitly uncertain."""

    candidate_index: int
    incumbent_id: str
    relation_type: MemoryRelationType | None
    direction: RelationDirection
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RevisionCompositionProof:
    """Transient proof that one candidate may revise one incumbent losslessly."""

    candidate_index: int
    incumbent_id: str
    same_memory_identity: bool
    preserves_incumbent_truth: bool
    candidate_is_canonical_composite: bool
    reason: str = ""

    @property
    def eligible(self) -> bool:
        """Candidate admission already proved the candidate's current Evidence complete."""

        return (
            self.same_memory_identity
            and self.preserves_incumbent_truth
            and self.candidate_is_canonical_composite
        )


class RecheckReading(str, Enum):
    """What one re-check reads."""

    # A claim without a read this revision reads its rebound Supports in the normal order.
    NORMAL_ORDER = "normal_order"
    # A read or unknown claim reads only the equivalent Candidates' current Evidence.
    CANDIDATE_EVIDENCE = "candidate_evidence"


@dataclass(frozen=True)
class SupportRecheckRequest:
    memory_id: str
    reading: RecheckReading
    # Equivalent Candidates whose current Evidence a CANDIDATE_EVIDENCE re-check reads.
    candidates: tuple[RawMemory, ...] = ()


@dataclass(frozen=True)
class Coordination:
    operations: tuple[ReconcileOperation, ...]
    # Candidates consumed without a decision by an unresolved component.
    unresolved_candidate_count: int = 0


class _Treatment(str, Enum):
    CONSUMED = "consumed"
    STAGED = "staged"
    REPLACES = "replaces"


@dataclass(frozen=True)
class _Ledger:
    candidates: Sequence[RawMemory]
    incumbents: Sequence[Memory]
    relations: Sequence[RelationLedgerEntry]
    proofs: Mapping[tuple[int, str], RevisionCompositionProof]
    supports: Mapping[str, MemorySupport]
    # Candidate/Memory pairs whose conflict a pending Review already holds: no re-check repeats it.
    rechecked_pairs: frozenset[tuple[int, str]]

    def edges(self, memory_id: str, relation_type: MemoryRelationType) -> list[RelationLedgerEntry]:
        return [
            entry for entry in self.relations
            if entry.incumbent_id == memory_id and entry.relation_type is relation_type
            and (relation_type is not MemoryRelationType.REFINES
                 or entry.direction is RelationDirection.CHALLENGER_TO_CANDIDATE)
        ]


def _ledger(
    *,
    candidates: Sequence[RawMemory],
    incumbents: Sequence[Memory],
    relations: Sequence[RelationLedgerEntry],
    proofs: Sequence[RevisionCompositionProof],
    supports: Mapping[str, MemorySupport],
    rechecked_pairs: frozenset[tuple[int, str]],
) -> _Ledger:
    incumbent_ids = {memory.id for memory in incumbents}
    pairs = {(entry.candidate_index, entry.incumbent_id) for entry in relations}
    if len(pairs) != len(relations) or any(
        not 0 <= index < len(candidates) or memory_id not in incumbent_ids for index, memory_id in pairs
    ):
        raise ReconciliationContractError(
            "relation_ledger_incomplete",
            "relation ledger contains duplicate or unknown candidate/incumbent references",
        )
    if set(supports) != incumbent_ids:
        raise ReconciliationContractError(
            "support_ledger_incomplete", "Support results do not cover every incumbent exactly once",
        )
    proofs_by_pair = {(proof.candidate_index, proof.incumbent_id): proof for proof in proofs}
    if len(proofs_by_pair) != len(proofs):
        raise ReconciliationContractError("duplicate_revision_proof", "duplicate revision composition proof")
    return _Ledger(candidates, incumbents, relations, proofs_by_pair, supports, rechecked_pairs)


def plan_rechecks(
    *,
    candidates: Sequence[RawMemory],
    incumbents: Sequence[Memory],
    relations: Sequence[RelationLedgerEntry],
    proofs: Sequence[RevisionCompositionProof] = (),
    supports: Mapping[str, MemorySupport],
    rechecked_pairs: frozenset[tuple[int, str]] = frozenset(),
) -> tuple[SupportRecheckRequest, ...]:
    """The single re-check each conflicting claim gets before the table decides."""
    ledger = _ledger(
        candidates=candidates, incumbents=incumbents, relations=relations, proofs=proofs, supports=supports,
        rechecked_pairs=rechecked_pairs,
    )
    _, unresolved = _unresolved_component(ledger.relations, _seeds(ledger))
    requests = []
    for memory in ledger.incumbents:
        support = ledger.supports[memory.id]
        if memory.id in unresolved or support.rechecked:
            continue
        equivalents = ledger.edges(memory.id, MemoryRelationType.EQUIVALENT)
        contradictions = ledger.edges(memory.id, MemoryRelationType.CONTRADICTS)
        if equivalents and contradictions:
            continue
        if support.result is SupportResult.UNAFFECTED and any(
            (entry.candidate_index, memory.id) not in ledger.rechecked_pairs for entry in contradictions
        ):
            requests.append(SupportRecheckRequest(memory.id, RecheckReading.NORMAL_ORDER))
        elif support.result in {SupportResult.UNSUPPORTED, SupportResult.UNRESOLVED_PARTIAL_COVERAGE}:
            fresh = tuple(
                ledger.candidates[entry.candidate_index] for entry in equivalents
                if (entry.candidate_index, memory.id) not in ledger.rechecked_pairs
            )
            if fresh:
                requests.append(SupportRecheckRequest(memory.id, RecheckReading.CANDIDATE_EVIDENCE, fresh))
    return tuple(requests)


def supported_refiners(
    *,
    candidates: Sequence[RawMemory],
    incumbents: Sequence[Memory],
    relations: Sequence[RelationLedgerEntry],
    proofs: Sequence[RevisionCompositionProof] = (),
    supports: Mapping[str, MemorySupport],
    rechecked_pairs: frozenset[tuple[int, str]] = frozenset(),
) -> dict[str, tuple[int, ...]]:
    """Candidates that refine each old Memory whose Support keeps it; only these may revise it.

    An old Memory in an unresolved component gets no decision this round, so its
    refinements need no comparison.
    """
    ledger = _ledger(
        candidates=candidates, incumbents=incumbents, relations=relations, proofs=proofs, supports=supports,
        rechecked_pairs=rechecked_pairs,
    )
    _, unresolved = _unresolved_component(ledger.relations, _seeds(ledger))
    grouped: dict[str, list[int]] = {}
    for entry in ledger.relations:
        if (
            entry.incumbent_id not in unresolved
            and ledger.supports[entry.incumbent_id].result in _KEEPS_SUPPORT
            and entry.relation_type is MemoryRelationType.REFINES
            and entry.direction is RelationDirection.CHALLENGER_TO_CANDIDATE
        ):
            grouped.setdefault(entry.incumbent_id, []).append(entry.candidate_index)
    return {memory_id: tuple(sorted(indices)) for memory_id, indices in grouped.items()}


def coordinate(
    *,
    candidates: Sequence[RawMemory],
    incumbents: Sequence[Memory],
    relations: Sequence[RelationLedgerEntry],
    proofs: Sequence[RevisionCompositionProof] = (),
    supports: Mapping[str, MemorySupport],
    rechecked_pairs: frozenset[tuple[int, str]] = frozenset(),
) -> Coordination:
    """Apply the combination table to final Support results and return one operation per claim."""
    ledger = _ledger(
        candidates=candidates, incumbents=incumbents, relations=relations, proofs=proofs, supports=supports,
        rechecked_pairs=rechecked_pairs,
    )
    seeds = _seeds(ledger)
    _, unresolved = _unresolved_component(ledger.relations, seeds)
    decided = {memory.id: _decide(ledger, memory) for memory in ledger.incumbents if memory.id not in unresolved}

    # A Candidate treated differently by different old Memories, or staged in the
    # Reviews of several, has no single decision: each approval would apply it anew.
    treatments: dict[int, list[_Treatment]] = {}
    for decision in decided.values():
        for index, treatment in decision.treatments:
            treatments.setdefault(index, []).append(treatment)
    divided = {
        index for index, kinds in treatments.items()
        if len(set(kinds)) > 1 or kinds.count(_Treatment.STAGED) > 1
    }
    if divided:
        seeds |= {
            (entry.candidate_index, entry.incumbent_id) for entry in ledger.relations
            if entry.candidate_index in divided and entry.relation_type is not MemoryRelationType.UNRELATED
        }
    component_candidates, unresolved = _unresolved_component(ledger.relations, seeds)

    consumed = set(component_candidates)
    operations: list[ReconcileOperation] = []
    for memory in ledger.incumbents:
        support = ledger.supports[memory.id]
        if memory.id in unresolved:
            reason = (
                support.reason if support.result is SupportResult.UNRESOLVED_CAPACITY
                else "Unresolved claim relationship; preserve existing Support and Evidence"
            )
            operations.append(replace(_kept_unchanged(reason), memory_id=memory.id))
            continue
        decision = decided[memory.id]
        consumed.update(index for index, _ in decision.treatments)
        operations.append(replace(decision.operation, memory_id=memory.id))
    additions = [
        ReconcileOperation(action=ReconcileAction.ADD, memory=raw, reason="independent current claim")
        for index, raw in enumerate(ledger.candidates)
        if index not in consumed
    ]
    return Coordination((*additions, *operations), len(component_candidates))


@dataclass(frozen=True)
class _Decision:
    operation: ReconcileOperation
    treatments: tuple[tuple[int, _Treatment], ...] = ()


def _decide(ledger: _Ledger, memory: Memory) -> _Decision:
    """One old Memory's row of the combination table."""
    support = ledger.supports[memory.id]
    equivalents = ledger.edges(memory.id, MemoryRelationType.EQUIVALENT)
    contradictions = ledger.edges(memory.id, MemoryRelationType.CONTRADICTS)
    refiners = ledger.edges(memory.id, MemoryRelationType.REFINES)
    related = [entry for entry in ledger.relations if entry.incumbent_id == memory.id]

    def candidate(index: int) -> RawMemory:
        return ledger.candidates[index]

    if support.result is SupportResult.UNRESOLVED_CAPACITY:
        # Reached only without related Candidates; related ones seed the unresolved component.
        return _Decision(_kept_unchanged(support.reason))

    if equivalents and contradictions:
        rejection_rebind = candidate(equivalents[0].candidate_index)
        return _Decision(
            replace(
                _kept(support),
                reviews=tuple(
                    CoordinatorReview(
                        candidate=candidate(entry.candidate_index),
                        proposal=CoordinatorProposal.SUPERSEDE,
                        reason=entry.reason or "the current source states an equivalent and a contradicting claim",
                        rejection_rebind=rejection_rebind,
                    )
                    for entry in contradictions
                ),
            ),
            (*_staged(contradictions), *_staged(equivalents)),
        )

    if contradictions and support.result is not SupportResult.UNSUPPORTED:
        # SUPPORTED, UNAFFECTED whose re-check could not run, or partial coverage.
        return _Decision(
            replace(
                _kept(support),
                reviews=tuple(
                    CoordinatorReview(
                        candidate=candidate(entry.candidate_index),
                        proposal=CoordinatorProposal.SUPERSEDE,
                        reason=entry.reason or "current claim contradicts the old Memory",
                    )
                    for entry in contradictions
                ),
            ),
            _staged(contradictions),
        )

    if support.result in _KEEPS_SUPPORT:
        if len(refiners) == 1:
            refiner = refiners[0]
            proof = ledger.proofs.get((refiner.candidate_index, memory.id))
            if proof is not None and proof.eligible:
                return _Decision(
                    ReconcileOperation(
                        action=ReconcileAction.UPDATE,
                        memory=candidate(refiner.candidate_index),
                        reason=proof.reason or refiner.reason or "lossless additive revision",
                    ),
                    (*_consumed(equivalents), (refiner.candidate_index, _Treatment.REPLACES)),
                )
        return _Decision(_kept(support, equivalents[0].reason if equivalents else None), _consumed(equivalents))

    if equivalents:
        # UNSUPPORTED or partial coverage, still not supported after its one re-check.
        return _Decision(
            replace(
                _kept(support),
                reviews=tuple(
                    CoordinatorReview(
                        candidate=candidate(entry.candidate_index),
                        proposal=CoordinatorProposal.REBIND,
                        reason=entry.reason or "an equivalent current claim is not confirmed by Support",
                    )
                    for entry in equivalents
                ),
            ),
            _staged(equivalents),
        )

    if support.result is SupportResult.UNRESOLVED_PARTIAL_COVERAGE:
        return _Decision(_kept_unchanged(support.reason))

    # UNSUPPORTED after the whole reading order.
    if len(contradictions) == 1 and len(related) == 1:
        challenger = contradictions[0]
        return _Decision(
            ReconcileOperation(
                action=ReconcileAction.SUPERSEDE,
                memory=candidate(challenger.candidate_index),
                reason=challenger.reason or support.reason or "current claim contradicts incumbent",
            ),
            ((challenger.candidate_index, _Treatment.REPLACES),),
        )
    # Several contradicting Candidates name no single successor: each stands on its own.
    return _Decision(ReconcileOperation(
        action=ReconcileAction.DELETE, reason=support.reason or "support removed",
    ))


def _kept(support: MemorySupport, reason: str | None = None) -> ReconcileOperation:
    """Keep the old Memory: rebound to its verified current Evidence, or unchanged without it."""
    if support.result in _KEEPS_SUPPORT:
        return ReconcileOperation(
            action=ReconcileAction.NOOP,
            memory=support.evidence[0] if support.evidence else None,
            reason=reason or support.reason or "current Source Unit support retained",
        )
    return _kept_unchanged(support.reason)


def _kept_unchanged(reason: str) -> ReconcileOperation:
    return ReconcileOperation(
        action=ReconcileAction.NOOP,
        reason=reason or "Support revalidation unresolved; preserve existing evidence",
        support_revalidation_skipped=True,
    )


def _consumed(entries: Sequence[RelationLedgerEntry]) -> tuple[tuple[int, _Treatment], ...]:
    return tuple((entry.candidate_index, _Treatment.CONSUMED) for entry in entries)


def _staged(entries: Sequence[RelationLedgerEntry]) -> tuple[tuple[int, _Treatment], ...]:
    return tuple((entry.candidate_index, _Treatment.STAGED) for entry in entries)


def _seeds(ledger: _Ledger) -> set[tuple[int, str]]:
    """Pairs that leave their related component unresolved.

    Explicitly uncertain pairs; every related pair of an ``UNRESOLVED(capacity)``
    old Memory; and a refinement whose proof preserves all of an UNSUPPORTED old
    Memory's truth, since the Candidate then restates knowledge Support rejected.
    """
    seeds = {(entry.candidate_index, entry.incumbent_id) for entry in ledger.relations if entry.relation_type is None}
    for entry in ledger.relations:
        result = ledger.supports[entry.incumbent_id].result
        if entry.relation_type is MemoryRelationType.UNRELATED:
            continue
        if result is SupportResult.UNRESOLVED_CAPACITY:
            seeds.add((entry.candidate_index, entry.incumbent_id))
        elif (
            result is SupportResult.UNSUPPORTED
            and entry.relation_type is MemoryRelationType.REFINES
            and entry.direction is RelationDirection.CHALLENGER_TO_CANDIDATE
            and (proof := ledger.proofs.get((entry.candidate_index, entry.incumbent_id))) is not None
            and proof.preserves_incumbent_truth
        ):
            seeds.add((entry.candidate_index, entry.incumbent_id))
    return seeds


def _unresolved_component(
    relations: Sequence[RelationLedgerEntry], seeds: set[tuple[int, str]],
) -> tuple[set[int], set[str]]:
    """Keep uncertainty local without letting a shared candidate escape as ADD.

    A candidate can touch more than one incumbent. The related component of every
    seed stays together; unrelated pairs never spread uncertainty to independent knowledge.
    """
    candidates = {candidate_index for candidate_index, _ in seeds}
    incumbents = {incumbent_id for _, incumbent_id in seeds}
    while True:
        size = len(candidates) + len(incumbents)
        for entry in relations:
            if entry.relation_type is not MemoryRelationType.UNRELATED and (
                entry.candidate_index in candidates or entry.incumbent_id in incumbents
            ):
                candidates.add(entry.candidate_index)
                incumbents.add(entry.incumbent_id)
        if len(candidates) + len(incumbents) == size:
            return candidates, incumbents
