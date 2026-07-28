"""Classify a complete Source Unit incumbent ledger in bounded LLM batches.

This module produces decisions only. It never mutates Memory lifecycle state;
the Lifecycle Planner validates the complete ledger and builds the atomic plan.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from memforge.llm.structured import StructuredLlmError
from memforge.models import Memory, RawMemory, ReconcileAction, ReconcileOperation

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "ReconciliationFailure",
    "ReconciliationMetrics",
    "ReconciliationResult",
    "reconcile_memories",
]


@dataclass(frozen=True)
class ReconciliationFailure:
    """Failure metadata for a reconciliation call that produced no safe decisions."""

    error_type: str
    error: str


@dataclass(frozen=True)
class ReconciliationMetrics:
    """Transport and latency measurements for one bounded reconciliation."""

    structured_llm_calls: int = 0
    model_batch_count: int = 0
    structured_llm_elapsed_ms: int = 0
    reconciliation_elapsed_ms: int = 0


@dataclass(frozen=True)
class ReconciliationResult:
    """Reconciliation call result with operations and optional failure metadata."""

    operations: list[ReconcileOperation]
    failure: ReconciliationFailure | None = None
    metrics: ReconciliationMetrics = ReconciliationMetrics()


RECONCILIATION_PROMPT = """You are reconciling team knowledge. A document was updated and new facts
were extracted. Compare them against existing memories from the same document
and the updated document content.

<update_mode>{update_mode}</update_mode>
<diff_stats>
{diff_stats}
</diff_stats>
<changed_hunks>
{changed_hunks}
</changed_hunks>

<ledger_contract>
{ledger_contract}
</ledger_contract>

For each new extraction listed for this phase, decide ONE action:

- ADD: Genuinely new information not covered by any existing memory.
- UPDATE: An existing memory covers the same durable claim, but the new evidence
  adds a real factual detail that must be present in the canonical text. Provide
  the merged current text. This is not a wording cleanup: the durable information
  carried by the Memory has increased. The old materialization will be preserved
  with a timestamp but hidden from default search.
- SUPERSEDE: An existing memory covers the same topic but is now materially wrong.
  The old fact was true before but is no longer. The new fact replaces the old meaning.
  The old memory will be preserved with a timestamp but hidden from search.
- DELETE: An existing memory is demonstrably false or was extracted in error.
- NOOP: The new extraction adds nothing beyond what existing memories capture.
  Use NOOP when it is semantically equivalent to an incumbent, including
  paraphrases, synonyms, reordered wording, stylistic rewrites, and extraction
  wording variance. Preserve the incumbent memory ID even when the source text
  or the newly extracted candidate is worded differently.

When the ledger contract requests an incumbent audit, audit EVERY existing memory
slot in the batch against the updated document and return exactly one explicit
decision in its corresponding response slot:
- NOOP when it remains supported or is disjoint from the changed evidence.
- DELETE when this source unit no longer supports it and there is no replacement.
Never omit an active slot. Missing or populated unused slots invalidate the
whole batch.

When update_mode is diff_guided, use changed_hunks as the authority for what
changed. Use the full updated document only to validate support and understand
context. Do not update, supersede, or delete memories from other documents.
In diff_guided mode, DELETE or SUPERSEDE an existing memory only when <changed_hunks>
removes, replaces, or contradicts the supporting text, or when the updated
document clearly shows the relevant section now states a different current fact.
Do not DELETE solely because support is absent from unrelated context or because
the updated document excerpt may be incomplete.

<doc_type>{doc_type}</doc_type>

<updated_document>
{updated_document}
</updated_document>

<new_extractions>
{new_extractions}
</new_extractions>

<existing_memories>
{existing_memories}
</existing_memories>

Rules:
1. Fill exactly the named response slots required by the ledger contract.
2. Never echo a candidate index or datastore memory ID. Identity is bound by the
   named response slot; matching an incumbent uses only its bounded incumbent_slot.
3. For UPDATE, SUPERSEDE, and NOOP candidate relations, specify the affected
   incumbent_slot.
4. For UPDATE, provide the merged current text as "updated_content".
5. If uncertain between UPDATE and SUPERSEDE, prefer SUPERSEDE when the old meaning is materially wrong.
6. If an existing memory has corroboration_count >= 3 and you want to SUPERSEDE it,
   set "flag_for_review": true.
7. For UPDATE updated_content and SUPERSEDE replacement memory content, write the
   canonical current memory, not the edit history. The replacement memory content must state the current durable fact as it should appear in search results.
   Do not write replacement content as edit history such as "no longer marked",
   "was removed", "the document changed", or "previously". Put that rationale in
   the reason field instead. Only mention a historical transition in memory
   content when the updated document itself says the transition is a durable fact.
8. Never use UPDATE only to improve phrasing or to mirror a semantically
   equivalent source rewrite. That is NOOP. UPDATE requires at least one durable
   factual detail in updated_content that the incumbent did not already carry.

Return ONLY the fixed-slot JSON object required by the response schema."""


RECONCILIATION_INCUMBENT_BATCH_SIZE = 30
RECONCILIATION_CANDIDATE_BATCH_SIZE = 24
RECONCILIATION_BATCH_VALIDATION_ATTEMPTS = 2

_CANDIDATE_RELATION_CONTRACT = """This is one cell of a complete candidate-by-incumbent
relation matrix:
- Return exactly one judgment in every active named candidate response slot.
- Return null in every unused candidate response slot.
- Compare each candidate only with the listed incumbents.
- Use ADD when the candidate does not match an incumbent in this cell; the deterministic
  reducer will authorize a global ADD only after every incumbent cell is complete.
- For NOOP, UPDATE, or SUPERSEDE, select one listed incumbent_slot.
- Do not emit unindexed incumbent lifecycle decisions and do not use DELETE."""

_INCUMBENT_AUDIT_CONTRACT = """This is the independent incumbent-support audit:
- There are no new extractions in this phase.
- Return exactly one NOOP or DELETE judgment in every active named incumbent response slot.
- Return null in every unused incumbent response slot.
- Decide only whether the updated Source Unit still supports the incumbent.
- Do not emit ADD, UPDATE, or SUPERSEDE."""


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
    """Classify new candidates and every incumbent, failing closed on ambiguity."""
    started = perf_counter()
    structured_llm_calls = 0
    structured_llm_elapsed_seconds = 0.0
    model_batch_count = 0

    def metrics() -> ReconciliationMetrics:
        return ReconciliationMetrics(
            structured_llm_calls=structured_llm_calls,
            model_batch_count=model_batch_count,
            structured_llm_elapsed_ms=max(
                0,
                round(structured_llm_elapsed_seconds * 1000),
            ),
            reconciliation_elapsed_ms=max(
                0,
                round((perf_counter() - started) * 1000),
            ),
        )

    if not new_extractions and not existing_memories:
        return _return_result([], metrics=metrics(), include_metadata=include_metadata)

    # If no existing memories, everything is ADD (skip LLM call)
    if not existing_memories:
        return _return_result(
            [ReconcileOperation(action=ReconcileAction.ADD, memory=raw) for raw in new_extractions],
            metrics=metrics(),
            include_metadata=include_metadata,
        )

    indexed_extractions = list(enumerate(new_extractions))
    candidate_batches = [
        indexed_extractions[offset : offset + RECONCILIATION_CANDIDATE_BATCH_SIZE]
        for offset in range(0, len(indexed_extractions), RECONCILIATION_CANDIDATE_BATCH_SIZE)
    ]

    try:
        decisions: list[dict] = []
        incumbent_batches = [
            existing_memories[offset : offset + RECONCILIATION_INCUMBENT_BATCH_SIZE]
            for offset in range(0, len(existing_memories), RECONCILIATION_INCUMBENT_BATCH_SIZE)
        ]
        # Every input size uses the same two phase-specific protocols. Candidate
        # and incumbent identity is bound by fixed response slots rather than
        # echoed by the model in an open list.
        for incumbent_batch in incumbent_batches:
            for candidate_batch in candidate_batches:
                batch_decisions, calls, elapsed = await _run_reconciliation_batch(
                    structured_llm_method=(structured_llm_client.reconcile_candidate_relations),
                    llm_model=llm_model,
                    prompt=_render_reconciliation_prompt(
                        ledger_contract=_CANDIDATE_RELATION_CONTRACT,
                        indexed_extractions=candidate_batch,
                        incumbents=incumbent_batch,
                        doc_type=doc_type,
                        updated_document=updated_document,
                        update_mode=update_mode,
                        changed_hunks=changed_hunks,
                        update_plan_stats=update_plan_stats,
                    ),
                    expected_indices=tuple(index for index, _ in candidate_batch),
                    incumbents=incumbent_batch,
                    require_incumbent_coverage=False,
                    allow_unindexed_incumbents=False,
                    decision_phase="candidate_relation",
                )
                model_batch_count += 1
                structured_llm_calls += calls
                structured_llm_elapsed_seconds += elapsed
                decisions.extend(batch_decisions)

            audit_decisions, calls, elapsed = await _run_reconciliation_batch(
                structured_llm_method=(structured_llm_client.audit_incumbent_support),
                llm_model=llm_model,
                prompt=_render_reconciliation_prompt(
                    ledger_contract=_INCUMBENT_AUDIT_CONTRACT,
                    indexed_extractions=[],
                    incumbents=incumbent_batch,
                    doc_type=doc_type,
                    updated_document=updated_document,
                    update_mode=update_mode,
                    changed_hunks=changed_hunks,
                    update_plan_stats=update_plan_stats,
                ),
                expected_indices=(),
                incumbents=incumbent_batch,
                require_incumbent_coverage=True,
                allow_unindexed_incumbents=True,
                incumbent_audit=True,
                decision_phase="incumbent_audit",
            )
            model_batch_count += 1
            structured_llm_calls += calls
            structured_llm_elapsed_seconds += elapsed
            decisions.extend(audit_decisions)

        return _return_result(
            _merge_complete_batch_decisions(decisions, new_extractions, existing_memories),
            metrics=metrics(),
            include_metadata=include_metadata,
        )

    except (StructuredLlmError, KeyError) as e:
        logger.warning("Structured reconciliation failed: %s — skipping reconciliation mutations", e)
        operations = [] if existing_memories else _fallback_add_all(new_extractions)
        return _return_result(
            operations,
            failure=ReconciliationFailure(error_type="structured_llm_error", error=str(e)),
            metrics=metrics(),
            include_metadata=include_metadata,
        )
    except Exception as e:
        logger.error("Reconciliation LLM call failed: %s — skipping reconciliation mutations", e)
        operations = [] if existing_memories else _fallback_add_all(new_extractions)
        return _return_result(
            operations,
            failure=ReconciliationFailure(error_type="unexpected_error", error=str(e)),
            metrics=metrics(),
            include_metadata=include_metadata,
        )


def _render_reconciliation_prompt(
    *,
    ledger_contract: str,
    indexed_extractions: list[tuple[int, RawMemory]],
    incumbents: list[Memory],
    doc_type: str,
    updated_document: str | None,
    update_mode: str,
    changed_hunks: str | None,
    update_plan_stats: dict | None,
) -> str:
    new_json = json.dumps(
        [
            {
                "slot": f"slot_{slot_index:02d}",
                "content": raw.content,
                "memory_type": raw.memory_type,
                "confidence": raw.confidence,
                "entity_refs": raw.entity_refs,
            }
            for slot_index, (_, raw) in enumerate(indexed_extractions)
        ],
        indent=2,
    )
    existing_json = json.dumps(
        [
            {
                "response_slot": f"slot_{slot_index:02d}",
                "incumbent_slot": slot_index,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "confidence": memory.confidence,
                "corroboration_count": memory.corroboration_count,
            }
            for slot_index, memory in enumerate(incumbents)
        ],
        indent=2,
    )
    return RECONCILIATION_PROMPT.format(
        ledger_contract=ledger_contract,
        update_mode=update_mode,
        diff_stats=json.dumps(update_plan_stats or {}, indent=2),
        changed_hunks=(changed_hunks or "")[:40_000],
        doc_type=doc_type,
        updated_document=(updated_document or "")[:100_000],
        new_extractions=new_json,
        existing_memories=existing_json,
    )


def _bind_candidate_relation_slots(
    slots,
    *,
    expected_indices: tuple[int, ...],
    incumbents: list[Memory],
) -> list[dict]:
    """Bind model judgments to candidate and incumbent identities owned by the request."""

    if len(slots) != RECONCILIATION_CANDIDATE_BATCH_SIZE:
        raise ValueError("candidate relation response does not contain the fixed protocol slots")
    decisions: list[dict] = []
    for slot_index, judgment in enumerate(slots):
        if slot_index >= len(expected_indices):
            if judgment is not None:
                raise ValueError(f"unused candidate relation slot {slot_index} must be null")
            continue
        if judgment is None:
            raise ValueError(f"candidate relation slot {slot_index} must not be null")
        data = judgment.model_dump()
        incumbent_slot = data.pop("incumbent_slot")
        action = str(data.get("action", "")).upper()
        if action == "ADD":
            if incumbent_slot is not None:
                raise ValueError(f"ADD candidate relation slot {slot_index} must not select an incumbent")
        else:
            if not isinstance(incumbent_slot, int) or incumbent_slot >= len(incumbents):
                raise ValueError(f"candidate relation slot {slot_index} selected an inactive incumbent slot")
            data["memory_id"] = incumbents[incumbent_slot].id
        if action == "UPDATE" and not data.get("updated_content"):
            raise ValueError(f"UPDATE candidate relation slot {slot_index} requires updated_content")
        data["index"] = expected_indices[slot_index]
        decisions.append(data)
    return decisions


def _bind_incumbent_audit_slots(
    slots,
    *,
    incumbents: list[Memory],
) -> list[dict]:
    """Bind incumbent support judgments to datastore identities by request order."""

    if len(slots) != RECONCILIATION_INCUMBENT_BATCH_SIZE:
        raise ValueError("incumbent audit response does not contain the fixed protocol slots")
    decisions: list[dict] = []
    for slot_index, judgment in enumerate(slots):
        if slot_index >= len(incumbents):
            if judgment is not None:
                raise ValueError(f"unused incumbent audit slot {slot_index} must be null")
            continue
        if judgment is None:
            raise ValueError(f"incumbent audit slot {slot_index} must not be null")
        decisions.append(
            {
                **judgment.model_dump(),
                "memory_id": incumbents[slot_index].id,
            }
        )
    return decisions


async def _run_reconciliation_batch(
    *,
    structured_llm_method,
    llm_model: str,
    prompt: str,
    expected_indices: tuple[int, ...],
    incumbents: list[Memory],
    require_incumbent_coverage: bool,
    allow_unindexed_incumbents: bool,
    decision_phase: str,
    incumbent_audit: bool = False,
) -> tuple[list[dict], int, float]:
    calls = 0
    elapsed_seconds = 0.0
    batch_decisions: list[dict] = []
    for validation_attempt in range(RECONCILIATION_BATCH_VALIDATION_ATTEMPTS):
        calls += 1
        llm_started = perf_counter()
        try:
            response = await structured_llm_method(
                prompt,
                max_tokens=4096,
                model=llm_model,
            )
        finally:
            elapsed_seconds += perf_counter() - llm_started
        try:
            if decision_phase == "candidate_relation":
                batch_decisions = _bind_candidate_relation_slots(
                    response.ordered_slots(),
                    expected_indices=expected_indices,
                    incumbents=incumbents,
                )
            elif decision_phase == "incumbent_audit":
                batch_decisions = _bind_incumbent_audit_slots(
                    response.ordered_slots(),
                    incumbents=incumbents,
                )
            else:
                raise ValueError(f"unknown reconciliation decision phase: {decision_phase}")
            _validate_complete_reconciliation_batch(
                batch_decisions,
                incumbents,
                expected_indices=set(expected_indices),
                require_incumbent_coverage=require_incumbent_coverage,
                allow_unindexed_incumbents=allow_unindexed_incumbents,
                incumbent_audit=incumbent_audit,
            )
        except ValueError as exc:
            if validation_attempt + 1 >= RECONCILIATION_BATCH_VALIDATION_ATTEMPTS:
                raise
            logger.warning(
                "Reconciliation batch validation failed: %s — retrying only this batch",
                exc,
            )
            prompt = (
                f"{prompt}\n\n<validation_feedback>\n"
                f"The previous response was rejected: {exc}. "
                "Return a complete corrected decisions ledger that satisfies every rule.\n"
                "</validation_feedback>"
            )
            continue
        break
    return batch_decisions, calls, elapsed_seconds


def _return_result(
    operations: list[ReconcileOperation],
    *,
    failure: ReconciliationFailure | None = None,
    metrics: ReconciliationMetrics,
    include_metadata: bool,
) -> list[ReconcileOperation] | ReconciliationResult:
    if include_metadata:
        return ReconciliationResult(
            operations=operations,
            failure=failure,
            metrics=metrics,
        )
    return operations


def _parse_decisions(
    decisions: list[dict],
    new_extractions: list[RawMemory],
    existing_memories: list[Memory],
    *,
    add_uncovered: bool = True,
) -> list[ReconcileOperation]:
    """Parse LLM decisions into ReconcileOperations."""
    ops: list[ReconcileOperation] = []
    existing_ids = {mem.id for mem in existing_memories}

    # Track which indices we've seen
    seen_indices: set[int] = set()

    for dec in decisions:
        idx = dec.get("index", -1)
        if not isinstance(idx, int) or idx < 0 or idx >= len(new_extractions) or idx in seen_indices:
            memory_id = dec.get("memory_id")
            if memory_id not in existing_ids:
                continue
            action_str = dec.get("action", "").upper()
            try:
                action = ReconcileAction(action_str)
            except ValueError:
                continue
            if action not in (ReconcileAction.DELETE, ReconcileAction.NOOP):
                continue
            ops.append(
                ReconcileOperation(
                    action=action,
                    memory_id=memory_id,
                    reason=dec.get("reason", ""),
                    flag_for_review=bool(dec.get("flag_for_review")),
                )
            )
            continue
        seen_indices.add(idx)

        action_str = dec.get("action", "ADD").upper()
        try:
            action = ReconcileAction(action_str)
        except ValueError:
            action = ReconcileAction.ADD

        raw = new_extractions[idx]
        memory_id = dec.get("memory_id")
        reason = dec.get("reason", "")

        if action == ReconcileAction.UPDATE and memory_id and dec.get("updated_content"):
            # UPDATE: create a modified version of the raw memory with merged content
            updated_raw = RawMemory(
                content=dec["updated_content"],
                memory_type=raw.memory_type,
                confidence=raw.confidence,
                entity_refs=raw.entity_refs,
                valid_from=raw.valid_from,
                valid_until=raw.valid_until,
                extraction_context=raw.extraction_context,
                evidence_quote=raw.evidence_quote,
                evidence_anchor=raw.evidence_anchor,
                source_observation_id=raw.source_observation_id,
                required_source_observation_ids=list(raw.required_source_observation_ids),
            )
            ops.append(
                ReconcileOperation(
                    action=action,
                    memory_id=memory_id,
                    memory=updated_raw,
                    reason=reason,
                    flag_for_review=bool(dec.get("flag_for_review")),
                )
            )
        elif action in (ReconcileAction.SUPERSEDE, ReconcileAction.DELETE) and memory_id:
            ops.append(
                ReconcileOperation(
                    action=action,
                    memory_id=memory_id,
                    memory=raw,
                    reason=reason,
                    flag_for_review=bool(dec.get("flag_for_review")),
                )
            )
        elif action == ReconcileAction.NOOP:
            ops.append(
                ReconcileOperation(
                    action=action,
                    memory_id=memory_id,
                    memory=raw,
                    reason=reason,
                    flag_for_review=bool(dec.get("flag_for_review")),
                )
            )
        else:
            # ADD or fallback
            ops.append(
                ReconcileOperation(
                    action=ReconcileAction.ADD,
                    memory=raw,
                    reason=reason,
                    flag_for_review=bool(dec.get("flag_for_review")),
                )
            )

    # Any new extractions not covered by decisions → ADD
    for i, raw in enumerate(new_extractions):
        if add_uncovered and i not in seen_indices:
            ops.append(
                ReconcileOperation(
                    action=ReconcileAction.ADD,
                    memory=raw,
                    reason="Not covered by reconciliation",
                )
            )

    return ops


def _validate_complete_reconciliation_batch(
    decisions: list[dict],
    incumbents: list[Memory],
    *,
    expected_indices: set[int],
    require_incumbent_coverage: bool,
    allow_unindexed_incumbents: bool,
    incumbent_audit: bool = False,
) -> None:
    indices = [
        decision.get("index")
        for decision in decisions
        if isinstance(decision.get("index"), int)
        and int(decision["index"]) in expected_indices
    ]
    unexpected_indices = sorted(
        {
            int(decision["index"])
            for decision in decisions
            if isinstance(decision.get("index"), int)
            and int(decision["index"]) not in expected_indices
        }
    )
    if unexpected_indices:
        raise ValueError(f"unexpected new extraction decisions: {unexpected_indices}")
    duplicate_indices = sorted({index for index in indices if indices.count(index) > 1})
    if duplicate_indices:
        raise ValueError(f"duplicate new extraction decisions: {duplicate_indices}")
    missing_indices = sorted(expected_indices.difference(indices))
    if missing_indices:
        raise ValueError(f"missing new extraction decisions: {missing_indices}")

    expected = {memory.id for memory in incumbents}
    unknown_incumbents = sorted(
        {
            str(decision["memory_id"])
            for decision in decisions
            if decision.get("memory_id") is not None
            and decision.get("memory_id") not in expected
        }
    )
    if unknown_incumbents:
        raise ValueError(f"unexpected incumbent decisions: {unknown_incumbents}")
    if not allow_unindexed_incumbents:
        unindexed = [
            str(decision.get("memory_id"))
            for decision in decisions
            if not isinstance(decision.get("index"), int)
        ]
        if unindexed:
            raise ValueError(
                f"candidate relation batch emitted unindexed incumbent decisions: {unindexed}"
            )
        deleted_candidates = sorted(
            int(decision["index"])
            for decision in decisions
            if str(decision.get("action", "")).upper() == "DELETE"
            and isinstance(decision.get("index"), int)
        )
        if deleted_candidates:
            raise ValueError(
                f"candidate relation batch emitted DELETE decisions: {deleted_candidates}"
            )
    seen = {
        str(decision["memory_id"])
        for decision in decisions
        if decision.get("memory_id") in expected
    }
    missing = sorted(expected.difference(seen)) if require_incumbent_coverage else []
    if missing:
        raise ValueError(f"missing incumbent decisions: {missing}")

    if incumbent_audit:
        invalid_actions = sorted(
            {
                str(decision.get("action", "")).upper()
                for decision in decisions
                if str(decision.get("action", "")).upper() not in {"NOOP", "DELETE"}
            }
        )
        if invalid_actions:
            raise ValueError(
                f"incumbent support audit emitted invalid actions: {invalid_actions}"
            )

    for memory_id in sorted(expected):
        group = [item for item in decisions if item.get("memory_id") == memory_id]
        if not group:
            continue
        dispositions = {_incumbent_disposition(item) for item in group}
        if None in dispositions:
            raise ValueError(f"invalid incumbent decision for {memory_id}")
        if len(dispositions) > 1:
            raise ValueError(f"conflicting incumbent decisions for {memory_id}")
        replacements = [
            item
            for item in group
            if isinstance(item.get("index"), int)
            and str(item.get("action", "")).upper() in {"UPDATE", "SUPERSEDE"}
        ]
        if "replace" in dispositions and not replacements:
            raise ValueError(
                f"replacement decision for incumbent {memory_id} requires a new extraction index"
            )
        if len(replacements) > 1:
            raise ValueError(f"multiple replacement candidates for incumbent {memory_id}")


def _incumbent_disposition(decision: dict) -> str | None:
    action = str(decision.get("action", "")).upper()
    if action == "NOOP":
        return "keep"
    if action in {"UPDATE", "SUPERSEDE"}:
        return "replace"
    if action == "DELETE":
        return "remove"
    return None


def _merge_complete_batch_decisions(
    decisions: list[dict],
    new_extractions: list[RawMemory],
    existing_memories: list[Memory],
) -> list[ReconcileOperation]:
    """Merge bounded incumbent batches into one unambiguous operation ledger."""

    existing_ids = {memory.id for memory in existing_memories}
    _validate_composed_incumbent_consistency(decisions, existing_ids=existing_ids)
    by_index: dict[int, list[dict]] = {index: [] for index in range(len(new_extractions))}
    by_incumbent: dict[str, list[dict]] = {memory_id: [] for memory_id in existing_ids}
    for decision in decisions:
        memory_id = decision.get("memory_id")
        if memory_id in existing_ids:
            by_incumbent[str(memory_id)].append(decision)
        index = decision.get("index")
        if isinstance(index, int) and index in by_index:
            by_index[index].append(decision)

    operations: list[ReconcileOperation] = []
    consumed_incumbents: set[str] = set()
    for index, raw in enumerate(new_extractions):
        candidates = by_index[index]
        destructive = [
            item
            for item in candidates
            if str(item.get("action", "")).upper() in {"UPDATE", "SUPERSEDE", "DELETE"}
            and item.get("memory_id") in existing_ids
        ]
        destructive_targets = {str(item["memory_id"]) for item in destructive}
        if len(destructive_targets) > 1:
            raise ValueError(
                f"new extraction {index} matches multiple destructive incumbents: "
                f"{sorted(destructive_targets)}"
            )
        if destructive:
            chosen = destructive[0]
        else:
            noop = [
                item
                for item in candidates
                if str(item.get("action", "")).upper() == "NOOP"
                and item.get("memory_id") in existing_ids
            ]
            chosen = (
                sorted(noop, key=lambda item: str(item.get("memory_id")))[0]
                if noop
                else next(
                    (
                        item
                        for item in candidates
                        if str(item.get("action", "")).upper() in {"ADD", "NOOP"}
                    ),
                    {"index": index, "action": "ADD", "reason": "new claim"},
                )
            )
        chosen = dict(chosen)
        chosen_memory_id = chosen.get("memory_id")
        if chosen_memory_id in consumed_incumbents:
            if str(chosen.get("action", "")).upper() != "NOOP":
                raise ValueError(
                    f"incumbent {chosen_memory_id} matches multiple destructive new extractions"
                )
            # The candidate is explicitly a duplicate, but the incumbent's one
            # lifecycle KEEP was already recorded by an earlier candidate.
            chosen["memory_id"] = None
        parsed = _parse_decisions(
            [chosen],
            new_extractions,
            existing_memories,
            add_uncovered=False,
        )
        if len(parsed) != 1:
            raise ValueError(f"new extraction {index} did not produce exactly one decision")
        operations.extend(parsed)
        if chosen.get("memory_id") in existing_ids:
            consumed_incumbents.add(str(chosen["memory_id"]))

    for memory in existing_memories:
        if memory.id in consumed_incumbents:
            continue
        group = by_incumbent[memory.id]
        unindexed = [item for item in group if not isinstance(item.get("index"), int)]
        decision = dict(unindexed[0] if unindexed else group[0])
        # Indexed NOOP rows are also explicit incumbent KEEP decisions. If the
        # candidate chose another compatible match, normalize this row to the
        # one incumbent-only operation required by the Lifecycle Planner.
        decision["index"] = None
        decision["memory_id"] = memory.id
        parsed = _parse_decisions(
            [decision],
            new_extractions,
            existing_memories,
            add_uncovered=False,
        )
        if len(parsed) != 1:
            raise ValueError(f"incumbent {memory.id} did not produce exactly one decision")
        operations.extend(parsed)

    return operations


def _validate_composed_incumbent_consistency(
    decisions: list[dict],
    *,
    existing_ids: set[str],
) -> None:
    """Require candidate relations and independent support audits to agree."""

    for memory_id in sorted(existing_ids):
        related = [
            decision
            for decision in decisions
            if decision.get("memory_id") == memory_id and isinstance(decision.get("index"), int)
        ]
        audits = [
            decision
            for decision in decisions
            if decision.get("memory_id") == memory_id and not isinstance(decision.get("index"), int)
        ]
        if len(audits) != 1:
            raise ValueError(f"incumbent {memory_id} requires exactly one independent support audit")
        relation_dispositions = {
            _incumbent_disposition(decision) for decision in related if _incumbent_disposition(decision) is not None
        }
        if len(relation_dispositions) > 1:
            raise ValueError(f"conflicting candidate relations for incumbent {memory_id}")
        if not relation_dispositions:
            continue
        relation_disposition = next(iter(relation_dispositions))
        audit_disposition = _incumbent_disposition(audits[0])
        compatible = (relation_disposition == "keep" and audit_disposition == "keep") or (
            relation_disposition == "replace" and audit_disposition == "remove"
        )
        if not compatible:
            raise ValueError(f"conflicting incumbent decisions for {memory_id}")


def _fallback_add_all(new_extractions: list[RawMemory]) -> list[ReconcileOperation]:
    """Treat candidates as ADD only when no incumbent lifecycle is at risk."""
    return [ReconcileOperation(action=ReconcileAction.ADD, memory=raw) for raw in new_extractions]
