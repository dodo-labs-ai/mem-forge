"""Join the Support line and the Relation line of one Source Unit revision.

SupportRelationCoordinator is program code. It runs after both lines finish and
combines, for every same-Unit old Memory, its Memory-level Support result with
the Relation edges that point at it. Neither line decides the other's truth.

A claim that could not be judged (``UNRESOLVED(capacity)`` or
``UNRESOLVED(invalid_response)``) is kept unchanged and its related Candidates
are consumed; every other claim takes its row:

   ======================================  ============  ==========================================
   Support                                 Relation      Action
   ======================================  ============  ==========================================
   SUPPORTED, or UNAFFECTED                none, equiv.  keep and rebind; equivalents consumed
   SUPPORTED                               contradicts   keep and rebind; the Candidate is added
   UNAFFECTED                              contradicts   re-check in the normal reading order
   UNSUPPORTED                             equivalent    re-check with the Candidate's Evidence
   UNSUPPORTED                             contradicts   SUPERSEDE
   UNSUPPORTED                             none          remove this source's Support
   UNRESOLVED(partial_coverage)            equivalent    re-check with the Candidate's Evidence
   UNRESOLVED(partial_coverage)            contradicts   keep unchanged; the Candidate is added
   UNRESOLVED(partial_coverage)            none          keep unchanged
   ======================================  ============  ==========================================

The table raises no Review. Two incompatible claims that both keep Support stay
Active, and relation discovery after commit records their relation as for any
two Memories. A claim is re-checked at most once per revision. A supported
re-check with a Candidate's Evidence keeps the old Memory, binds it to that
Evidence and consumes the Candidate; a claim still unsupported keeps its own
result, so an UNSUPPORTED one loses this source's Support and the Candidate is
added, and a partially covered one stays unchanged with the Candidate consumed.
A normal-order re-check yields an ordinary Support result that enters the table
again. REFINES and uncertain edges keep the local unresolved relationship
rules; a refinement revises only an old Memory no Candidate contradicts. A
Candidate that receives different treatments across old Memories leaves its
whole related component unresolved: consumed this round, with no ADD and no
destructive action. An unresolved relationship withholds only what a Candidate
would do. An old Memory of the component whose own Support result keeps it is
still bound to its verified current Evidence; the others stay unchanged. A
Candidate the Relation line could not judge even alone is consumed the same
way; its missing row leaves the Relation line incomplete, so
DestructiveValidation withholds every destructive decision of the Unit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType

from memforge.memory.evidence import RelationDirection
from memforge.memory.relation_classifier import SameUnitRelationType
from memforge.models import (
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
    "revisions_to_verify",
    "supported_refiners",
]


class ReconciliationContractError(ValueError):
    """A bounded fail-closed reconciliation invariant violation."""

    retryable = False

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
    # The model's output for one ReadingGroup alone stays invalid after its correction.
    UNRESOLVED_INVALID_RESPONSE = "unresolved_invalid_response"
    # A prior Evidence part is UNKNOWN under partial projection coverage.
    UNRESOLVED_PARTIAL_COVERAGE = "unresolved_partial_coverage"

    @property
    def unresolved(self) -> bool:
        return self in UNRESOLVED_RESULTS.values()

    @property
    def unjudged(self) -> bool:
        """The claim could not be judged: one ReadingGroup alone exceeds capacity or gets invalid output."""
        return self in {SupportResult.UNRESOLVED_CAPACITY, SupportResult.UNRESOLVED_INVALID_RESPONSE}


# A Support Assessment's unresolved reason and the Memory-level result it yields, in precedence order.
UNRESOLVED_RESULTS: Mapping[str, SupportResult] = MappingProxyType({
    "capacity": SupportResult.UNRESOLVED_CAPACITY,
    "invalid_response": SupportResult.UNRESOLVED_INVALID_RESPONSE,
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
    # This revision already re-checked the claim; the table takes this result as final.
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
        unresolved = UNRESOLVED_RESULTS.get(assessment.unresolved or "")
        if unresolved is not None and unresolved.unjudged:
            return MemorySupport(
                result=unresolved, reason=assessment.reason, rechecked=True, assessments=(assessment,),
            )
        return replace(self, rechecked=True)


def memory_support(assessments: Sequence[SupportAssessment], *, rechecked: bool = False) -> MemorySupport:
    """Combine one old Memory's Support assessments by the table's precedence.

    Any ``UNRESOLVED(capacity)`` Support decides the Memory, then any
    ``UNRESOLVED(invalid_response)`` one, then any ``UNRESOLVED(partial_coverage)`` one. Otherwise a read that found Support
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
    relation_type: SameUnitRelationType | None
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
        """All three model judgments authorize a lossless revision; none is a formal proof."""

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
    # Candidates consumed without a decision by an unresolved component, unjudged ones included.
    unresolved_candidate_count: int = 0


class _Treatment(str, Enum):
    CONSUMED = "consumed"
    REPLACES = "replaces"


@dataclass(frozen=True)
class _Ledger:
    candidates: Sequence[RawMemory]
    incumbents: Sequence[Memory]
    relations: Sequence[RelationLedgerEntry]
    proofs: Mapping[tuple[int, str], RevisionCompositionProof]
    supports: Mapping[str, MemorySupport]
    # Candidates the Relation line could not judge even alone.
    unjudged_candidates: frozenset[int] = frozenset()
    # Revisions whose Refinement verification found a fact of the old claim missing.
    unverified_revisions: frozenset[tuple[int, str]] = frozenset()

    def edges(self, memory_id: str, relation_type: SameUnitRelationType) -> list[RelationLedgerEntry]:
        return [
            entry for entry in self.relations
            if entry.incumbent_id == memory_id and entry.relation_type is relation_type
            and (relation_type is not SameUnitRelationType.REFINES
                 or entry.direction is RelationDirection.CHALLENGER_TO_CANDIDATE)
        ]


def _ledger(
    *,
    candidates: Sequence[RawMemory],
    incumbents: Sequence[Memory],
    relations: Sequence[RelationLedgerEntry],
    proofs: Sequence[RevisionCompositionProof],
    supports: Mapping[str, MemorySupport],
    unjudged_candidates: frozenset[int],
    unverified_revisions: frozenset[tuple[int, str]] = frozenset(),
) -> _Ledger:
    incumbent_ids = {memory.id for memory in incumbents}
    pairs = {(entry.candidate_index, entry.incumbent_id) for entry in relations}
    if len(pairs) != len(relations) or any(
        not 0 <= index < len(candidates) or memory_id not in incumbent_ids for index, memory_id in pairs
    ) or any(not 0 <= index < len(candidates) for index in unjudged_candidates):
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
    return _Ledger(candidates, incumbents, relations, proofs_by_pair, supports, unjudged_candidates, unverified_revisions)


def plan_rechecks(
    *,
    candidates: Sequence[RawMemory],
    incumbents: Sequence[Memory],
    relations: Sequence[RelationLedgerEntry],
    proofs: Sequence[RevisionCompositionProof] = (),
    supports: Mapping[str, MemorySupport],
    unjudged_candidates: frozenset[int] = frozenset(),
) -> tuple[SupportRecheckRequest, ...]:
    """The single re-check each conflicting claim gets before the table decides."""
    ledger = _ledger(
        candidates=candidates, incumbents=incumbents, relations=relations, proofs=proofs, supports=supports,
        unjudged_candidates=unjudged_candidates,
    )
    _, unresolved = _unresolved_component(ledger, _seeds(ledger))
    requests = []
    for memory in ledger.incumbents:
        support = ledger.supports[memory.id]
        if memory.id in unresolved or support.rechecked:
            continue
        equivalents = ledger.edges(memory.id, SameUnitRelationType.EQUIVALENT)
        if support.result is SupportResult.UNAFFECTED and ledger.edges(memory.id, SameUnitRelationType.CONTRADICTS):
            requests.append(SupportRecheckRequest(memory.id, RecheckReading.NORMAL_ORDER))
        elif equivalents and support.result in {SupportResult.UNSUPPORTED, SupportResult.UNRESOLVED_PARTIAL_COVERAGE}:
            requests.append(SupportRecheckRequest(
                memory.id, RecheckReading.CANDIDATE_EVIDENCE,
                tuple(ledger.candidates[entry.candidate_index] for entry in equivalents),
            ))
    return tuple(requests)


def supported_refiners(
    *,
    candidates: Sequence[RawMemory],
    incumbents: Sequence[Memory],
    relations: Sequence[RelationLedgerEntry],
    proofs: Sequence[RevisionCompositionProof] = (),
    supports: Mapping[str, MemorySupport],
    unjudged_candidates: frozenset[int] = frozenset(),
) -> dict[str, tuple[int, ...]]:
    """Candidates that refine each old Memory the table may revise: kept by its Support and not contradicted.

    An old Memory in an unresolved component gets no decision this round, and a
    contradicted one is never revised, so their refinements need no comparison.
    """
    ledger = _ledger(
        candidates=candidates, incumbents=incumbents, relations=relations, proofs=proofs, supports=supports,
        unjudged_candidates=unjudged_candidates,
    )
    _, unresolved = _unresolved_component(ledger, _seeds(ledger))
    grouped: dict[str, list[int]] = {}
    for entry in ledger.relations:
        if (
            entry.incumbent_id not in unresolved
            and ledger.supports[entry.incumbent_id].result in _KEEPS_SUPPORT
            and entry.relation_type is SameUnitRelationType.REFINES
            and entry.direction is RelationDirection.CHALLENGER_TO_CANDIDATE
            and not ledger.edges(entry.incumbent_id, SameUnitRelationType.CONTRADICTS)
        ):
            grouped.setdefault(entry.incumbent_id, []).append(entry.candidate_index)
    return {memory_id: tuple(sorted(indices)) for memory_id, indices in grouped.items()}


def revisions_to_verify(
    *,
    candidates: Sequence[RawMemory],
    incumbents: Sequence[Memory],
    relations: Sequence[RelationLedgerEntry],
    proofs: Sequence[RevisionCompositionProof] = (),
    supports: Mapping[str, MemorySupport],
    unjudged_candidates: frozenset[int] = frozenset(),
) -> tuple[tuple[int, str], ...]:
    """Each (Candidate index, old Memory id) the table would revise, before Refinement verification answers.

    The table revises an old Memory whose final Support result keeps it, that no
    Candidate contradicts and that exactly one Candidate refines with an
    eligible proof, unless the old Memory is in an unresolved component. Every
    such pair is verified, also one whose Candidate a later divided treatment
    might hold back, so no revision is released unverified.
    """
    ledger = _ledger(
        candidates=candidates, incumbents=incumbents, relations=relations, proofs=proofs, supports=supports,
        unjudged_candidates=unjudged_candidates,
    )
    _, unresolved = _unresolved_component(ledger, _seeds(ledger))
    pairs = []
    for memory in ledger.incumbents:
        refiners = ledger.edges(memory.id, SameUnitRelationType.REFINES)
        if (
            memory.id in unresolved
            or ledger.supports[memory.id].result not in _KEEPS_SUPPORT
            or ledger.edges(memory.id, SameUnitRelationType.CONTRADICTS)
            or len(refiners) != 1
        ):
            continue
        proof = ledger.proofs.get((refiners[0].candidate_index, memory.id))
        if proof is not None and proof.eligible:
            pairs.append((refiners[0].candidate_index, memory.id))
    return tuple(pairs)


def coordinate(
    *,
    candidates: Sequence[RawMemory],
    incumbents: Sequence[Memory],
    relations: Sequence[RelationLedgerEntry],
    proofs: Sequence[RevisionCompositionProof] = (),
    supports: Mapping[str, MemorySupport],
    unjudged_candidates: frozenset[int] = frozenset(),
    unverified_revisions: frozenset[tuple[int, str]] = frozenset(),
) -> Coordination:
    """Apply the combination table to final Support results and return one operation per claim.

    ``unverified_revisions`` are the revisions Refinement verification answered
    ``drops`` for: their old Memory is kept and the Candidate added, as with an
    ineligible proof.
    """
    ledger = _ledger(
        candidates=candidates, incumbents=incumbents, relations=relations, proofs=proofs, supports=supports,
        unjudged_candidates=unjudged_candidates, unverified_revisions=unverified_revisions,
    )
    seeds = _seeds(ledger)
    _, unresolved = _unresolved_component(ledger, seeds)
    decided = {memory.id: _decide(ledger, memory) for memory in ledger.incumbents if memory.id not in unresolved}

    # A Candidate treated differently by different old Memories has no single decision.
    treatments: dict[int, set[_Treatment]] = {}
    for decision in decided.values():
        for index, treatment in decision.treatments:
            treatments.setdefault(index, set()).add(treatment)
    divided = {index for index, kinds in treatments.items() if len(kinds) > 1}
    if divided:
        seeds |= {
            (entry.candidate_index, entry.incumbent_id) for entry in ledger.relations
            if entry.candidate_index in divided and entry.relation_type is not SameUnitRelationType.UNRELATED
        }
    component_candidates, unresolved = _unresolved_component(ledger, seeds)

    consumed = set(component_candidates)
    operations: list[ReconcileOperation] = []
    for memory in ledger.incumbents:
        support = ledger.supports[memory.id]
        if memory.id in unresolved:
            operations.append(replace(_kept_undecided(support), memory_id=memory.id))
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
    equivalents = ledger.edges(memory.id, SameUnitRelationType.EQUIVALENT)
    contradictions = ledger.edges(memory.id, SameUnitRelationType.CONTRADICTS)
    refiners = ledger.edges(memory.id, SameUnitRelationType.REFINES)
    related = [entry for entry in ledger.relations if entry.incumbent_id == memory.id]

    def candidate(index: int) -> RawMemory:
        return ledger.candidates[index]

    if support.result.unjudged:
        # Reached only without related Candidates; related ones seed the unresolved component.
        return _Decision(_kept_unchanged(support.reason))

    if support.result in _KEEPS_SUPPORT:
        # A contradicting Candidate is added beside the kept old Memory; discovery relates the two.
        if len(refiners) == 1 and not contradictions:
            refiner = refiners[0]
            proof = ledger.proofs.get((refiner.candidate_index, memory.id))
            if (
                proof is not None and proof.eligible
                and (refiner.candidate_index, memory.id) not in ledger.unverified_revisions
            ):
                return _Decision(
                    ReconcileOperation(
                        action=ReconcileAction.UPDATE,
                        memory=candidate(refiner.candidate_index),
                        reason=proof.reason or refiner.reason or "lossless additive revision",
                    ),
                    (*_consumed(equivalents), (refiner.candidate_index, _Treatment.REPLACES)),
                )
        return _Decision(_kept(support, equivalents[0].reason if equivalents else None), _consumed(equivalents))

    if support.result is SupportResult.UNRESOLVED_PARTIAL_COVERAGE:
        # Its unknown part keeps the old Memory; an equivalent restates it, a contradiction is added.
        return _Decision(_kept_unchanged(support.reason), _consumed(equivalents))

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


def _kept_undecided(support: MemorySupport) -> ReconcileOperation:
    """Keep an old Memory of an unresolved component; its own Support result still applies."""
    if support.result in _KEEPS_SUPPORT and support.evidence:
        return _kept(support, "Unresolved claim relationship; keep on verified current Evidence")
    return _kept_unchanged(
        support.reason if support.result.unjudged
        else "Unresolved claim relationship; preserve existing Support and Evidence"
    )


def _kept_unchanged(reason: str) -> ReconcileOperation:
    return ReconcileOperation(
        action=ReconcileAction.NOOP,
        reason=reason or "Support revalidation unresolved; preserve existing evidence",
        support_revalidation_skipped=True,
    )


def _consumed(entries: Sequence[RelationLedgerEntry]) -> tuple[tuple[int, _Treatment], ...]:
    return tuple((entry.candidate_index, _Treatment.CONSUMED) for entry in entries)


def _seeds(ledger: _Ledger) -> set[tuple[int, str]]:
    """Pairs that leave their related component unresolved.

    Explicitly uncertain pairs; every related pair of an old Memory whose claim
    could not be judged (``UNRESOLVED(capacity)`` or ``UNRESOLVED(invalid_response)``); and a refinement whose proof preserves all of an UNSUPPORTED old
    Memory's truth, since the Candidate then restates knowledge Support rejected.
    """
    seeds = {(entry.candidate_index, entry.incumbent_id) for entry in ledger.relations if entry.relation_type is None}
    for entry in ledger.relations:
        result = ledger.supports[entry.incumbent_id].result
        if entry.relation_type is SameUnitRelationType.UNRELATED:
            continue
        if result.unjudged:
            seeds.add((entry.candidate_index, entry.incumbent_id))
        elif (
            result is SupportResult.UNSUPPORTED
            and entry.relation_type is SameUnitRelationType.REFINES
            and entry.direction is RelationDirection.CHALLENGER_TO_CANDIDATE
            and (proof := ledger.proofs.get((entry.candidate_index, entry.incumbent_id))) is not None
            and proof.preserves_incumbent_truth
        ):
            seeds.add((entry.candidate_index, entry.incumbent_id))
    return seeds


def _unresolved_component(ledger: _Ledger, seeds: set[tuple[int, str]]) -> tuple[set[int], set[str]]:
    """Keep uncertainty local without letting a shared candidate escape as ADD.

    A candidate can touch more than one incumbent. The related component of every
    seed and of every unjudged Candidate stays together; unrelated pairs never
    spread uncertainty to independent knowledge.
    """
    relations = ledger.relations
    candidates = {candidate_index for candidate_index, _ in seeds} | ledger.unjudged_candidates
    incumbents = {incumbent_id for _, incumbent_id in seeds}
    while True:
        size = len(candidates) + len(incumbents)
        for entry in relations:
            if entry.relation_type is not SameUnitRelationType.UNRELATED and (
                entry.candidate_index in candidates or entry.incumbent_id in incumbents
            ):
                candidates.add(entry.candidate_index)
                incumbents.add(entry.incumbent_id)
        if len(candidates) + len(incumbents) == size:
            return candidates, incumbents
