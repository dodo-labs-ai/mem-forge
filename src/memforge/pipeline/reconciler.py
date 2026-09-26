"""The Relation line of one Source Unit revision, and its join with the Support line.

The model classifies facts only: candidate/incumbent relations and, only for
REFINES, revision eligibility. The Relation line never sees a Support result;
SupportRelationCoordinator combines the two lines in program code. Nothing here
mutates durable lifecycle state.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from time import perf_counter

from memforge.derivation_work import DerivationWorkStore
from memforge.evals.agent_evaluation import QualitySignal
from memforge.llm.structured import StructuredLlmError, StructuredLlmMetricsCollector, structured_llm_line_scope
from memforge.memory.evidence import RelationDirection
from memforge.memory.relation_classifier import (
    MemoryPair,
    MemoryPairClassificationError,
    MemoryRelationType,
    StructuredMemoryPairClassifier,
)
from memforge.models import (
    Memory,
    RawMemory,
    ReconcileOperation,
    content_hash,
    parse_memory_validity_date,
)
from memforge.pipeline.support_relation_coordinator import (
    MemorySupport,
    RelationLedgerEntry,
    RevisionCompositionProof,
    SupportRecheckRequest,
    coordinate,
    plan_rechecks,
    supported_refiners,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RelationLine",
    "ReconciliationFailure",
    "ReconciliationMetrics",
    "ReconciliationResult",
    "RecheckSupports",
    "assess_relations",
    "join_support_and_relation",
    "reconcile_memories",
]

# Runs SupportRelationCoordinator's re-checks and returns each claim's new Memory-level result.
RecheckSupports = Callable[[tuple[SupportRecheckRequest, ...]], Awaitable[Mapping[str, MemorySupport]]]


@dataclass(frozen=True, slots=True)
class ReconciliationFailure:
    """Failure metadata for a reconciliation that produced no safe ledger."""

    error_type: str
    reason_code: str
    error: str
    operation: str | None = None
    terminal_category: str | None = None
    error_code: str | None = None
    validation_fields: tuple[tuple[str, str], ...] = ()
    diagnostic: QualitySignal | None = None


class ReconciliationContractError(ValueError):
    """A bounded fail-closed reconciliation invariant violation."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class ReconciliationMetrics:
    """Model work and latency of the Relation line and its join; Support's calls are counted apart."""

    structured_llm_calls: int = 0
    model_batch_count: int = 0
    structured_llm_elapsed_ms: int = 0
    reconciliation_elapsed_ms: int = 0
    relation_pair_count: int = 0
    relation_prompt_chars: int = 0
    revision_proof_count: int = 0
    revision_proof_failure_count: int = 0


@dataclass(frozen=True, slots=True)
class RelationLine:
    """The Relation line's complete ledger for one revision, or the failure that left none."""

    entries: tuple[RelationLedgerEntry, ...] = ()
    proofs: tuple[RevisionCompositionProof, ...] = ()
    failure: ReconciliationFailure | None = None
    metrics: ReconciliationMetrics = ReconciliationMetrics()
    work_ids: tuple[str, ...] = ()
    # Candidates with a completion row, and the old-Memory catalog those rows cover.
    completed_candidate_count: int = 0
    catalog: frozenset[str] = frozenset()

    def covers(self, candidate_count: int, incumbent_ids: frozenset[str]) -> bool:
        """Whether every one of ``candidate_count`` Candidates has its row over exactly these old Memories."""
        return self.failure is None and self.completed_candidate_count == candidate_count and self.catalog == incumbent_ids


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Reconciliation result with operations and optional failure metadata."""

    operations: list[ReconcileOperation]
    failure: ReconciliationFailure | None = None
    metrics: ReconciliationMetrics = ReconciliationMetrics()
    work_ids: tuple[str, ...] = ()
    # SupportRelationCoordinator's re-checks; not executed when the caller supplies no re-check.
    rechecks: tuple[SupportRecheckRequest, ...] = ()
    # Candidates consumed without a decision by an unresolved component.
    unresolved_candidate_count: int = 0


async def assess_relations(
    new_extractions: list[RawMemory],
    existing_memories: list[Memory],
    *,
    structured_llm_client,
    llm_model: str,
    images: tuple = (),
    image_loader=None,
    work_store: DerivationWorkStore | None = None,
    derivation_id: str | None = None,
    operation_input_hash: str | None = None,
) -> RelationLine:
    """Judge every admitted Candidate against every same-Unit old Memory, without Support.

    A model or contract failure is returned, not raised: the revision is not
    committed, and nothing here may fall back to independent ADD.
    """
    started = perf_counter()
    pair_count = len(new_extractions) * len(existing_memories)
    catalog = frozenset(memory.id for memory in existing_memories)
    if not new_extractions or not existing_memories:
        # Nothing to relate: every Candidate's row is empty by definition.
        return RelationLine(
            metrics=ReconciliationMetrics(relation_pair_count=pair_count),
            completed_candidate_count=len(new_extractions), catalog=catalog,
        )
    from memforge.pipeline.claim_revision import assess_claim_pairs

    with structured_llm_line_scope() as line:
        try:
            assessed = await assess_claim_pairs(
                candidates=new_extractions, incumbents=existing_memories, client=structured_llm_client,
                model=llm_model, images=images, image_loader=image_loader,
                store=work_store, derivation_id=derivation_id, operation_input_hash=operation_input_hash,
            )
        except Exception as error:  # noqa: BLE001 - classified by _failure
            return RelationLine(
                failure=_failure(error, "assess_claim_revisions"),
                metrics=_add_calls(ReconciliationMetrics(relation_pair_count=pair_count), line, started),
            )
    entries = []
    proofs = []
    for index, incumbent_id, decision in assessed.decisions:
        relation = decision.relation
        if decision.status == "insufficient":
            entries.append(RelationLedgerEntry(
                candidate_index=index, incumbent_id=incumbent_id, relation_type=None,
                direction=RelationDirection.SYMMETRIC, reason=decision.reason,
            ))
            continue
        assert relation is not None
        entries.append(RelationLedgerEntry(
            candidate_index=index, incumbent_id=incumbent_id,
            relation_type=MemoryRelationType(relation.classification),
            direction=RelationDirection(relation.direction), reason=relation.reason,
        ))
        assessment = decision.revision_assessment
        if (relation.classification == "refines"
                and relation.direction == "challenger_to_candidate"
                and assessment is not None):
            proofs.append(RevisionCompositionProof(
                candidate_index=index, incumbent_id=incumbent_id,
                same_memory_identity=assessment.same_knowledge_item,
                preserves_incumbent_truth=assessment.preserves_incumbent_truth,
                candidate_is_canonical_composite=assessment.challenger_is_complete_current_claim,
                reason=decision.reason,
            ))
    metrics = ReconciliationMetrics(
        relation_pair_count=pair_count, relation_prompt_chars=assessed.prompt_chars, revision_proof_count=len(proofs),
    )
    return RelationLine(
        entries=tuple(entries), proofs=tuple(proofs), work_ids=assessed.work_ids,
        metrics=_add_calls(metrics, line, started),
        completed_candidate_count=assessed.completed_candidate_count, catalog=catalog,
    )


async def join_support_and_relation(
    relation: RelationLine,
    *,
    new_extractions: Sequence[RawMemory],
    existing_memories: Sequence[Memory],
    supports: Mapping[str, MemorySupport],
    structured_llm_client,
    llm_model: str,
    recheck: RecheckSupports | None = None,
    rechecked_pairs: frozenset[tuple[int, str]] = frozenset(),
) -> ReconciliationResult:
    """Run SupportRelationCoordinator over both finished lines.

    Each conflicting claim's single re-check runs first; a re-check execution
    failure raises like any Support failure. Several refinements of one kept old
    Memory are then compared pairwise, because only one of them may revise it.
    Without ``recheck`` the re-checks are reported and the claims keep their results.
    """
    started = perf_counter()
    assert relation.failure is None
    entries = list(relation.entries)
    coordinated = dict(
        candidates=new_extractions, incumbents=existing_memories, proofs=relation.proofs,
        rechecked_pairs=rechecked_pairs,
    )
    rechecks = plan_rechecks(relations=entries, supports=supports, **coordinated)
    if rechecks and recheck is not None:
        supports = {**supports, **await recheck(rechecks)}
    metrics = relation.metrics

    def result(operations=(), failure: ReconciliationFailure | None = None, unresolved: int = 0):
        return ReconciliationResult(
            operations=list(operations), failure=failure, work_ids=relation.work_ids, rechecks=rechecks,
            unresolved_candidate_count=unresolved, metrics=_add_calls(metrics, line, started),
        )

    transient_candidates = tuple(_transient_candidate(index, raw) for index, raw in enumerate(new_extractions))
    conditional_pairs = tuple(
        MemoryPair(challenger=transient_candidates[left], candidate=transient_candidates[right])
        for indices in supported_refiners(entries, supports).values()
        if len(indices) > 1
        for offset, left in enumerate(indices)
        for right in indices[offset + 1 :]
    )
    operation = "classify_memory_relations"
    with structured_llm_line_scope() as line:
        try:
            if conditional_pairs:
                classifier = StructuredMemoryPairClassifier(client=structured_llm_client, model=llm_model)
                conditional = await classifier.classify(conditional_pairs)
                metrics = replace(
                    metrics,
                    relation_pair_count=metrics.relation_pair_count + len(conditional_pairs),
                    relation_prompt_chars=metrics.relation_prompt_chars + conditional.prompt_chars,
                )
                entries = _without_conflicting_refinements(entries, conditional.decisions, transient_candidates)
            operation = "coordinate_support_and_relation"
            coordination = coordinate(relations=entries, supports=supports, **coordinated)
        except Exception as error:  # noqa: BLE001 - classified by _failure
            return result(failure=_failure(error, operation))
    return result(coordination.operations, unresolved=coordination.unresolved_candidate_count)


def _without_conflicting_refinements(
    entries: list[RelationLedgerEntry], decisions, transient_candidates: tuple[Memory, ...],
) -> list[RelationLedgerEntry]:
    """Refinements of one old Memory that contradict each other leave their edges uncertain."""
    conflicting_ids = {
        memory_id for decision in decisions
        if decision.relation_type is MemoryRelationType.CONTRADICTS
        for memory_id in decision.pair.key
    }
    conflicting_candidates = {index for index, candidate in enumerate(transient_candidates)
                              if candidate.id in conflicting_ids}
    return [
        replace(entry, relation_type=None, reason="Current refinement candidates conflict")
        if entry.candidate_index in conflicting_candidates
        and entry.relation_type is not MemoryRelationType.UNRELATED else entry
        for entry in entries
    ]


def _add_calls(
    metrics: ReconciliationMetrics, line: StructuredLlmMetricsCollector, started: float,
) -> ReconciliationMetrics:
    """Add one line's own model calls and elapsed time."""
    summary = line.summary(source_unit_elapsed_ms=0)
    return replace(
        metrics,
        structured_llm_calls=metrics.structured_llm_calls + summary.logical_calls,
        model_batch_count=metrics.model_batch_count + summary.logical_calls,
        structured_llm_elapsed_ms=metrics.structured_llm_elapsed_ms + summary.llm_elapsed_ms,
        reconciliation_elapsed_ms=metrics.reconciliation_elapsed_ms + _elapsed_ms(started),
    )


async def reconcile_memories(
    new_extractions: list[RawMemory],
    existing_memories: list[Memory],
    structured_llm_client,
    llm_model: str,
    *,
    supports: Mapping[str, MemorySupport],
    recheck: RecheckSupports | None = None,
    images: tuple = (),
    image_loader=None,
    work_store: DerivationWorkStore | None = None,
    derivation_id: str | None = None,
    operation_input_hash: str | None = None,
) -> ReconciliationResult:
    """Run the Relation line against Support results that are already known, then join them.

    Production runs the two lines concurrently; this sequential form serves
    callers whose Support results are pinned, such as offline replay.
    """
    relation = await assess_relations(
        new_extractions, existing_memories, structured_llm_client=structured_llm_client, llm_model=llm_model,
        images=images, image_loader=image_loader, work_store=work_store, derivation_id=derivation_id,
        operation_input_hash=operation_input_hash,
    )
    if relation.failure is not None:
        return ReconciliationResult(operations=[], failure=relation.failure, metrics=relation.metrics)
    return await join_support_and_relation(
        relation, new_extractions=new_extractions, existing_memories=existing_memories, supports=supports,
        structured_llm_client=structured_llm_client, llm_model=llm_model, recheck=recheck,
    )


def _failure(error: Exception, operation: str) -> ReconciliationFailure:
    """Fail closed: the revision is not committed and no Candidate falls back to ADD."""
    if isinstance(error, ReconciliationContractError):
        logger.warning("Relation-first reconciliation failed closed: %s", error)
        error_type, reason_code = "relation_first_error", error.reason_code
    elif isinstance(error, (StructuredLlmError, MemoryPairClassificationError, KeyError, ValueError)):
        logger.warning("Relation-first reconciliation failed closed: %s", error)
        error_type, reason_code = "relation_first_error", "relation_first_failed"
    else:
        logger.exception("Unexpected relation-first reconciliation failure")
        error_type, reason_code = "unexpected_error", "unexpected_reconciliation_failure"
    return ReconciliationFailure(
        error_type=error_type,
        reason_code=reason_code,
        error=str(error),
        operation=operation,
        terminal_category=getattr(error, "terminal_category", None),
        error_code=getattr(error, "error_code", None),
        validation_fields=getattr(error, "validation_fields", ()),
        diagnostic=getattr(error, "diagnostic", None),
    )


def _elapsed_ms(started: float) -> int:
    return max(0, round((perf_counter() - started) * 1000))


def _transient_candidate(index: int, raw: RawMemory) -> Memory:
    return Memory(
        id=f"candidate:{index}",
        memory_type=raw.memory_type,
        content=raw.content,
        content_hash=content_hash(raw.content),
        entity_refs=list(raw.entity_refs),
        confidence=raw.confidence,
        valid_from=parse_memory_validity_date(raw.valid_from),
        valid_until=parse_memory_validity_date(raw.valid_until),
    )
