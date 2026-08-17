"""Derive lifecycle operations from semantic relations and Source Unit support.

The model classifies facts only: exact candidate/incumbent relations, current
support, and (only for REFINES) revision eligibility. This module owns the
deterministic action matrix and never mutates durable lifecycle state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from time import perf_counter

from memforge.llm.structured import StructuredLlmError
from memforge.memory.evidence import RelationDirection
from memforge.memory.relation_classifier import (
    MemoryPair,
    MemoryPairClassificationError,
    MemoryPairDecision,
    MemoryRelationType,
    StructuredMemoryPairClassifier,
)
from memforge.models import (
    Memory,
    RawMemory,
    ReconcileAction,
    ReconcileOperation,
    content_hash,
    parse_memory_validity_date,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ReconciliationFailure",
    "ReconciliationMetrics",
    "ReconciliationResult",
    "RelationLedgerEntry",
    "RevisionCompositionProof",
    "SupportAuditEntry",
    "reconcile_memories",
    "reduce_relation_ledger",
]

RECONCILIATION_INCUMBENT_BATCH_SIZE = 30
REVISION_COMPOSITION_BATCH_SIZE = 64
RECONCILIATION_BATCH_VALIDATION_ATTEMPTS = 2

INCUMBENT_SUPPORT_AUDIT_PROMPT = """Audit whether the current Source Unit still supports every incumbent Memory.

This is a factual support judgment, not a lifecycle action. Return supported=true
when the exact claim remains entailed by the current Source Unit or is provably
disjoint from the changed evidence. Return supported=false only when the current
Source Unit removed, replaced, or contradicts the claim. Do not decide KEEP,
DELETE, UPDATE, or SUPERSEDE.

When update_mode is diff_guided, changed_hunks are authoritative for what
changed. Absence from an incomplete excerpt is not proof of unsupported status.
Return exactly one ordered decision for every listed incumbent.

<update_mode>{update_mode}</update_mode>
<diff_stats>{diff_stats}</diff_stats>
<changed_hunks>{changed_hunks}</changed_hunks>
<doc_type>{doc_type}</doc_type>
<updated_document>{updated_document}</updated_document>
<incumbents>{incumbents}</incumbents>
"""

REVISION_COMPOSITION_PROMPT = """Decide whether each exact REFINES pair may become one automatic Memory revision.

This proof is stricter than REFINES. Return true for all four booleans only when:
1. both texts are successive materializations of the same durable Memory identity;
2. the challenger preserves every durable truth carried by the incumbent while
   adding a compatible material detail (it does not merely narrow the old claim);
3. the challenger text itself is a complete canonical current claim, so it can
   be stored verbatim without model-written merging or historical narration.
4. the supplied exact current Primary Evidence excerpt entails the whole
   challenger claim, including the incumbent truth and the added detail.

Memory type and validity bounds are part of the claim. A change in modality or
validity must be justified as part of the same identity and truth-preservation
proof; otherwise return false.

If current Primary Evidence is empty, supports only the added detail, or the
claim depends on Required Evidence that is not supplied, set
current_evidence_entails_candidate=false.

Return false when the challenger is a sibling scenario, a narrower independent
claim, or would need text from the incumbent copied into a synthesized merge.
Do not return lifecycle actions or rewritten content. Return every pair_index
exactly once.

<refinement_pairs>{pairs_json}</refinement_pairs>
"""


@dataclass(frozen=True, slots=True)
class ReconciliationFailure:
    """Failure metadata for a reconciliation that produced no safe ledger."""

    error_type: str
    reason_code: str
    error: str


class ReconciliationContractError(ValueError):
    """A bounded fail-closed reconciliation invariant violation."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class ReconciliationMetrics:
    """Transport and latency measurements for one bounded reconciliation."""

    structured_llm_calls: int = 0
    model_batch_count: int = 0
    structured_llm_elapsed_ms: int = 0
    reconciliation_elapsed_ms: int = 0
    relation_pair_count: int = 0
    relation_prompt_chars: int = 0
    revision_proof_count: int = 0
    revision_proof_failure_count: int = 0


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Reconciliation result with operations and optional failure metadata."""

    operations: list[ReconcileOperation]
    failure: ReconciliationFailure | None = None
    metrics: ReconciliationMetrics = ReconciliationMetrics()


@dataclass(frozen=True, slots=True)
class RelationLedgerEntry:
    """One complete, datastore-bound candidate/incumbent relation."""

    candidate_index: int
    incumbent_id: str
    relation_type: MemoryRelationType
    direction: RelationDirection
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SupportAuditEntry:
    """One current Source Unit support judgment for an incumbent."""

    incumbent_id: str
    supported: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RevisionCompositionProof:
    """Transient proof that one candidate may revise one incumbent losslessly."""

    candidate_index: int
    incumbent_id: str
    same_memory_identity: bool
    preserves_incumbent_truth: bool
    candidate_is_canonical_composite: bool
    current_evidence_entails_candidate: bool
    complete_current_evidence: bool
    reason: str = ""

    @property
    def eligible(self) -> bool:
        return (
            self.same_memory_identity
            and self.preserves_incumbent_truth
            and self.candidate_is_canonical_composite
            and self.current_evidence_entails_candidate
            and self.complete_current_evidence
        )


async def reconcile_memories(
    new_extractions: list[RawMemory],
    existing_memories: list[Memory],
    doc_type: str,
    structured_llm_client,
    llm_model: str = "claude-sonnet-4-20250514",
    updated_document: str | None = None,
    update_mode: str = "full_document",
    changed_hunks: str | None = None,
    update_plan_stats: dict | None = None,
    include_metadata: bool = False,
) -> list[ReconcileOperation] | ReconciliationResult:
    """Classify a complete relation/support ledger and reduce it deterministically."""

    started = perf_counter()
    structured_llm_calls = 0
    model_batch_count = 0
    structured_llm_elapsed_seconds = 0.0
    relation_pair_count = 0
    relation_prompt_chars = 0
    revision_proof_count = 0
    revision_proof_failure_count = 0

    def metrics() -> ReconciliationMetrics:
        return ReconciliationMetrics(
            structured_llm_calls=structured_llm_calls,
            model_batch_count=model_batch_count,
            structured_llm_elapsed_ms=max(0, round(structured_llm_elapsed_seconds * 1000)),
            reconciliation_elapsed_ms=max(0, round((perf_counter() - started) * 1000)),
            relation_pair_count=relation_pair_count,
            relation_prompt_chars=relation_prompt_chars,
            revision_proof_count=revision_proof_count,
            revision_proof_failure_count=revision_proof_failure_count,
        )

    if not new_extractions and not existing_memories:
        return _return_result([], metrics=metrics(), include_metadata=include_metadata)
    if not existing_memories:
        return _return_result(
            [ReconcileOperation(action=ReconcileAction.ADD, memory=raw) for raw in new_extractions],
            metrics=metrics(),
            include_metadata=include_metadata,
        )

    try:
        classifier = StructuredMemoryPairClassifier(client=structured_llm_client, model=llm_model)
        transient_candidates = tuple(_transient_candidate(index, raw) for index, raw in enumerate(new_extractions))
        pairs = tuple(
            MemoryPair(challenger=candidate, candidate=incumbent)
            for candidate in transient_candidates
            for incumbent in existing_memories
        )
        relation_started = perf_counter()
        classification = await classifier.classify(pairs)
        structured_llm_elapsed_seconds += perf_counter() - relation_started
        structured_llm_calls += classification.llm_calls
        model_batch_count += classification.llm_calls
        relation_pair_count += len(pairs)
        relation_prompt_chars += classification.prompt_chars
        relation_entries = _bind_relation_entries(
            classification.decisions,
            candidate_count=len(new_extractions),
            incumbents=existing_memories,
        )

        audits, calls, elapsed = await _audit_incumbent_support(
            incumbents=existing_memories,
            structured_llm_client=structured_llm_client,
            llm_model=llm_model,
            doc_type=doc_type,
            updated_document=updated_document,
            update_mode=update_mode,
            changed_hunks=changed_hunks,
            update_plan_stats=update_plan_stats,
        )
        structured_llm_calls += calls
        model_batch_count += calls
        structured_llm_elapsed_seconds += elapsed

        refiners_by_incumbent = _supported_revision_candidates(relation_entries, audits)
        conditional_pairs = tuple(
            MemoryPair(challenger=transient_candidates[left], candidate=transient_candidates[right])
            for indices in refiners_by_incumbent.values()
            if len(indices) > 1
            for offset, left in enumerate(indices)
            for right in indices[offset + 1 :]
        )
        if conditional_pairs:
            conditional_started = perf_counter()
            conditional = await classifier.classify(conditional_pairs)
            structured_llm_elapsed_seconds += perf_counter() - conditional_started
            structured_llm_calls += conditional.llm_calls
            model_batch_count += conditional.llm_calls
            relation_pair_count += len(conditional_pairs)
            relation_prompt_chars += conditional.prompt_chars
            if any(decision.relation_type is MemoryRelationType.CONTRADICTS for decision in conditional.decisions):
                raise ReconciliationContractError(
                    "non_unique_refinement_conflict",
                    "multiple refinement candidates contain incompatible current assertions",
                )

        proof_requests = [
            (indices[0], incumbent_id)
            for incumbent_id, indices in refiners_by_incumbent.items()
            if len(indices) == 1
        ]
        proofs, calls, elapsed = await _prove_revision_compositions(
            requests=proof_requests,
            new_extractions=new_extractions,
            incumbents={memory.id: memory for memory in existing_memories},
            structured_llm_client=structured_llm_client,
            llm_model=llm_model,
        )
        revision_proof_count = len(proofs)
        revision_proof_failure_count = len(proof_requests) - len(proofs)
        structured_llm_calls += calls
        model_batch_count += calls
        structured_llm_elapsed_seconds += elapsed

        operations = reduce_relation_ledger(
            new_extractions=new_extractions,
            existing_memories=existing_memories,
            relations=relation_entries,
            support_audits=audits,
            revision_proofs=proofs,
        )
        return _return_result(operations, metrics=metrics(), include_metadata=include_metadata)
    except ReconciliationContractError as error:
        logger.warning("Relation-first reconciliation failed closed: %s", error)
        return _return_result(
            [],
            failure=ReconciliationFailure(
                error_type="relation_first_error",
                reason_code=error.reason_code,
                error=str(error),
            ),
            metrics=metrics(),
            include_metadata=include_metadata,
        )
    except (StructuredLlmError, MemoryPairClassificationError, KeyError, ValueError) as error:
        logger.warning("Relation-first reconciliation failed closed: %s", error)
        return _return_result(
            [],
            failure=ReconciliationFailure(
                error_type="relation_first_error",
                reason_code="relation_first_failed",
                error=str(error),
            ),
            metrics=metrics(),
            include_metadata=include_metadata,
        )
    except Exception as error:  # pragma: no cover - defensive provider boundary
        logger.exception("Unexpected relation-first reconciliation failure")
        return _return_result(
            [],
            failure=ReconciliationFailure(
                error_type="unexpected_error",
                reason_code="unexpected_reconciliation_failure",
                error=str(error),
            ),
            metrics=metrics(),
            include_metadata=include_metadata,
        )


def reduce_relation_ledger(
    *,
    new_extractions: list[RawMemory],
    existing_memories: list[Memory],
    relations: list[RelationLedgerEntry],
    support_audits: list[SupportAuditEntry],
    revision_proofs: list[RevisionCompositionProof] | None = None,
) -> list[ReconcileOperation]:
    """Apply the complete relation/support matrix to one deterministic action table."""

    incumbent_ids = {memory.id for memory in existing_memories}
    expected_pairs = {
        (candidate_index, memory.id)
        for candidate_index in range(len(new_extractions))
        for memory in existing_memories
    }
    actual_pairs = {(entry.candidate_index, entry.incumbent_id) for entry in relations}
    if len(actual_pairs) != len(relations) or actual_pairs != expected_pairs:
        raise ReconciliationContractError(
            "relation_ledger_incomplete",
            "relation ledger does not cover every exact candidate/incumbent pair once",
        )
    audits_by_id = {entry.incumbent_id: entry for entry in support_audits}
    if len(audits_by_id) != len(support_audits) or set(audits_by_id) != incumbent_ids:
        raise ReconciliationContractError(
            "support_ledger_incomplete",
            "support audit does not cover every incumbent exactly once",
        )
    proofs = revision_proofs or []
    proofs_by_pair = {(proof.candidate_index, proof.incumbent_id): proof for proof in proofs}
    if len(proofs_by_pair) != len(proofs):
        raise ReconciliationContractError(
            "duplicate_revision_proof",
            "duplicate revision composition proof",
        )

    by_incumbent: dict[str, list[RelationLedgerEntry]] = {memory_id: [] for memory_id in incumbent_ids}
    for entry in relations:
        by_incumbent[entry.incumbent_id].append(entry)

    consumed_candidates: set[int] = set()
    incumbent_operations: list[ReconcileOperation] = []
    for incumbent in existing_memories:
        audit = audits_by_id[incumbent.id]
        entries = by_incumbent[incumbent.id]
        equivalents = [entry for entry in entries if entry.relation_type is MemoryRelationType.EQUIVALENT]
        contradictions = [entry for entry in entries if entry.relation_type is MemoryRelationType.CONTRADICTS]
        refiners = [
            entry
            for entry in entries
            if entry.relation_type is MemoryRelationType.REFINES
            and entry.direction is RelationDirection.CHALLENGER_TO_CANDIDATE
        ]
        related = [entry for entry in entries if entry.relation_type is not MemoryRelationType.UNRELATED]

        if len(contradictions) == 1 and len(related) == 1:
            challenger = contradictions[0]
            consumed_candidates.add(challenger.candidate_index)
            incumbent_operations.append(
                ReconcileOperation(
                    action=ReconcileAction.SUPERSEDE,
                    memory_id=incumbent.id,
                    memory=new_extractions[challenger.candidate_index],
                    reason=challenger.reason or audit.reason or "current claim contradicts incumbent",
                    flag_for_review=audit.supported,
                )
            )
            continue

        if equivalents and not audit.supported:
            consumed_candidates.update(entry.candidate_index for entry in equivalents)
            incumbent_operations.append(
                ReconcileOperation(
                    action=ReconcileAction.DELETE,
                    memory_id=incumbent.id,
                    reason=(
                        "semantic equivalence conflicts with unsupported audit: "
                        f"{audit.reason or equivalents[0].reason}"
                    ),
                    flag_for_review=True,
                )
            )
            continue

        if audit.supported and len(refiners) == 1:
            refiner = refiners[0]
            proof = proofs_by_pair.get((refiner.candidate_index, incumbent.id))
            if proof is not None and proof.eligible:
                consumed_candidates.update(entry.candidate_index for entry in equivalents)
                consumed_candidates.add(refiner.candidate_index)
                incumbent_operations.append(
                    ReconcileOperation(
                        action=ReconcileAction.UPDATE,
                        memory_id=incumbent.id,
                        memory=new_extractions[refiner.candidate_index],
                        reason=proof.reason or refiner.reason or "lossless additive revision",
                    )
                )
                continue

        if equivalents and audit.supported:
            consumed_candidates.update(entry.candidate_index for entry in equivalents)
            selected = equivalents[0]
            incumbent_operations.append(
                ReconcileOperation(
                    action=ReconcileAction.NOOP,
                    memory_id=incumbent.id,
                    memory=new_extractions[selected.candidate_index],
                    reason=selected.reason or audit.reason or "equivalent current claim",
                )
            )
            continue

        incumbent_operations.append(
            ReconcileOperation(
                action=ReconcileAction.NOOP if audit.supported else ReconcileAction.DELETE,
                memory_id=incumbent.id,
                reason=audit.reason or ("current Source Unit support retained" if audit.supported else "support removed"),
            )
        )

    candidate_operations = [
        ReconcileOperation(action=ReconcileAction.ADD, memory=raw, reason="independent current claim")
        for index, raw in enumerate(new_extractions)
        if index not in consumed_candidates
    ]
    return [*candidate_operations, *incumbent_operations]


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


def _bind_relation_entries(
    decisions: tuple[MemoryPairDecision, ...],
    *,
    candidate_count: int,
    incumbents: list[Memory],
) -> list[RelationLedgerEntry]:
    incumbent_ids = {memory.id for memory in incumbents}
    entries: list[RelationLedgerEntry] = []
    for decision in decisions:
        challenger_id = decision.pair.challenger.id
        if not challenger_id.startswith("candidate:"):
            raise ReconciliationContractError(
                "relation_response_unknown_challenger",
                "relation classifier returned an unknown challenger",
            )
        candidate_index = int(challenger_id.split(":", 1)[1])
        if not 0 <= candidate_index < candidate_count or decision.pair.candidate.id not in incumbent_ids:
            raise ReconciliationContractError(
                "relation_response_out_of_scope",
                "relation classifier returned an out-of-scope pair",
            )
        entries.append(
            RelationLedgerEntry(
                candidate_index=candidate_index,
                incumbent_id=decision.pair.candidate.id,
                relation_type=decision.relation_type,
                direction=decision.direction,
                reason=decision.reason,
            )
        )
    return entries


async def _audit_incumbent_support(
    *,
    incumbents: list[Memory],
    structured_llm_client,
    llm_model: str,
    doc_type: str,
    updated_document: str | None,
    update_mode: str,
    changed_hunks: str | None,
    update_plan_stats: dict | None,
) -> tuple[list[SupportAuditEntry], int, float]:
    results: list[SupportAuditEntry] = []
    calls = 0
    elapsed = 0.0
    for offset in range(0, len(incumbents), RECONCILIATION_INCUMBENT_BATCH_SIZE):
        batch = incumbents[offset : offset + RECONCILIATION_INCUMBENT_BATCH_SIZE]
        prompt = INCUMBENT_SUPPORT_AUDIT_PROMPT.format(
            update_mode=update_mode,
            diff_stats=json.dumps(update_plan_stats or {}, ensure_ascii=False),
            changed_hunks=(changed_hunks or "")[:40_000],
            doc_type=doc_type,
            updated_document=(updated_document or "")[:100_000],
            incumbents=json.dumps(
                [
                    {"request_position": index, "content": memory.content, "memory_type": memory.memory_type}
                    for index, memory in enumerate(batch)
                ],
                ensure_ascii=False,
            ),
        )
        for attempt in range(RECONCILIATION_BATCH_VALIDATION_ATTEMPTS):
            calls += 1
            call_started = perf_counter()
            try:
                response = await structured_llm_client.audit_incumbent_support(
                    prompt,
                    max_tokens=4096,
                    model=llm_model,
                )
            finally:
                elapsed += perf_counter() - call_started
            if len(response.decisions) == len(batch):
                break
            if attempt + 1 == RECONCILIATION_BATCH_VALIDATION_ATTEMPTS:
                raise ReconciliationContractError(
                    "support_response_incomplete",
                    f"incumbent support response count {len(response.decisions)} "
                    f"does not match expected count {len(batch)}"
                )
            prompt += "\nReturn the complete ordered support ledger; the previous response count was invalid."
        results.extend(
            SupportAuditEntry(incumbent_id=memory.id, supported=bool(decision.supported), reason=decision.reason)
            for memory, decision in zip(batch, response.decisions, strict=True)
        )
    return results, calls, elapsed


def _supported_revision_candidates(
    relations: list[RelationLedgerEntry],
    audits: list[SupportAuditEntry],
) -> dict[str, tuple[int, ...]]:
    supported = {entry.incumbent_id for entry in audits if entry.supported}
    grouped: dict[str, list[int]] = {}
    for entry in relations:
        if (
            entry.incumbent_id in supported
            and entry.relation_type is MemoryRelationType.REFINES
            and entry.direction is RelationDirection.CHALLENGER_TO_CANDIDATE
        ):
            grouped.setdefault(entry.incumbent_id, []).append(entry.candidate_index)
    return {memory_id: tuple(sorted(indices)) for memory_id, indices in grouped.items()}


async def _prove_revision_compositions(
    *,
    requests: list[tuple[int, str]],
    new_extractions: list[RawMemory],
    incumbents: dict[str, Memory],
    structured_llm_client,
    llm_model: str,
) -> tuple[list[RevisionCompositionProof], int, float]:
    if not requests:
        return [], 0, 0.0
    method = getattr(structured_llm_client, "prove_revision_compositions", None)
    if not callable(method):
        return [], 0, 0.0
    proofs: list[RevisionCompositionProof] = []
    calls = 0
    elapsed = 0.0
    for offset in range(0, len(requests), REVISION_COMPOSITION_BATCH_SIZE):
        batch = requests[offset : offset + REVISION_COMPOSITION_BATCH_SIZE]
        prompt = REVISION_COMPOSITION_PROMPT.format(
            pairs_json=json.dumps(
                [
                    {
                        "pair_index": pair_index,
                        "incumbent": {
                            "content": incumbents[incumbent_id].content,
                            "memory_type": incumbents[incumbent_id].memory_type,
                            "valid_from": (
                                incumbents[incumbent_id].valid_from.isoformat()
                                if incumbents[incumbent_id].valid_from
                                else None
                            ),
                            "valid_until": (
                                incumbents[incumbent_id].valid_until.isoformat()
                                if incumbents[incumbent_id].valid_until
                                else None
                            ),
                        },
                        "challenger": {
                            "content": new_extractions[candidate_index].content,
                            "memory_type": new_extractions[candidate_index].memory_type,
                            "valid_from": new_extractions[candidate_index].valid_from,
                            "valid_until": new_extractions[candidate_index].valid_until,
                        },
                        "current_primary_evidence_excerpt": (
                            new_extractions[candidate_index].evidence_quote or ""
                        ),
                        "required_evidence_count": len(
                            new_extractions[candidate_index].required_source_observation_ids
                        ),
                    }
                    for pair_index, (candidate_index, incumbent_id) in enumerate(batch)
                ],
                ensure_ascii=False,
            )
        )
        by_index = None
        for attempt in range(RECONCILIATION_BATCH_VALIDATION_ATTEMPTS):
            calls += 1
            call_started = perf_counter()
            try:
                response = await method(prompt, max_tokens=4096, model=llm_model)
            except Exception as error:  # proof failure is a non-destructive fallback
                logger.warning(
                    "Revision composition proof failed; falling back to KEEP + ADD: %s",
                    error,
                )
                break
            finally:
                elapsed += perf_counter() - call_started
            candidate_by_index = {decision.pair_index: decision for decision in response.decisions}
            if (
                len(candidate_by_index) == len(response.decisions) == len(batch)
                and set(candidate_by_index) == set(range(len(batch)))
            ):
                by_index = candidate_by_index
                break
            if attempt + 1 == RECONCILIATION_BATCH_VALIDATION_ATTEMPTS:
                logger.warning(
                    "Revision composition coverage invalid; falling back to KEEP + ADD"
                )
                break
            prompt += "\nReturn every requested pair_index exactly once; the previous ledger was incomplete."
        if by_index is None:
            continue
        for pair_index, (candidate_index, incumbent_id) in enumerate(batch):
            raw = new_extractions[candidate_index]
            decision = by_index[pair_index]
            proofs.append(
                RevisionCompositionProof(
                    candidate_index=candidate_index,
                    incumbent_id=incumbent_id,
                    same_memory_identity=decision.same_memory_identity,
                    preserves_incumbent_truth=decision.preserves_incumbent_truth,
                    candidate_is_canonical_composite=decision.candidate_is_canonical_composite,
                    current_evidence_entails_candidate=(
                        decision.current_evidence_entails_candidate
                    ),
                    complete_current_evidence=bool(
                        raw.source_observation_id
                        and raw.evidence_resolved_from_block
                        and (raw.evidence_quote or "").strip()
                        and not raw.required_source_observation_ids
                    ),
                    reason=decision.reason,
                )
            )
    return proofs, calls, elapsed


def _return_result(
    operations: list[ReconcileOperation],
    *,
    failure: ReconciliationFailure | None = None,
    metrics: ReconciliationMetrics,
    include_metadata: bool,
) -> list[ReconcileOperation] | ReconciliationResult:
    if include_metadata:
        return ReconciliationResult(operations=operations, failure=failure, metrics=metrics)
    return operations
