"""Cross-document relation classification.

After Memories are inserted from a document sync, classifies their relationship
to access-compatible Memories from other documents. Bounded entity, semantic,
and lexical channels supply candidates; structured classification may produce
a conflict, refinement, clarification, or unrelated result.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from memforge.config import DEFAULT_ENRICHMENT_MAX_TOKENS
from memforge.llm.structured import StructuredLlmError
from memforge.memory.evidence import (
    AuthorityCase,
    CandidateBucket,
    EvidenceRelationRecord,
    EvidenceRole,
    EvidenceUnit,
    LifecycleAction,
    RelationOutcomeBundle,
    RelationRunRecord,
    RelationType,
    ReviewCase,
    build_candidate_universe,
    relation_run_id_for,
)
from memforge.memory.relation_candidate_retrieval import (
    CrossDocumentCandidateRetriever,
    CrossDocumentCandidateSelection,
    StaleCandidateSelectionError,
)
from memforge.models import (
    Memory,
    MemoryReview,
    ReviewKind,
    ReviewStatus,
    generate_deterministic_review_id,
)

if TYPE_CHECKING:
    from memforge.memory.store import MemoryStore
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = ["detect_cross_doc_contradictions"]

CONTRADICTION_LLM_BATCH_SIZE = 20
MAX_CONTRADICTION_PROMPT_CHARS = 120_000
MAX_CONTRADICTION_MEMORY_CONTENT_CHARS = 4_000


CONTRADICTION_PROMPT = """You are classifying how pairs of team knowledge memories relate.
These memories come from different source documents about the same entities.

For each pair, classify the relationship:
- CONTRADICTION: they make mutually incompatible claims about the same thing
- TEMPORAL: they describe the same subject or property at different time points
- CLARIFICATION: one adds compatible detail to the other without conflicting
- UNRELATED: they happen to share an entity but discuss different aspects

TEMPORAL and CLARIFICATION identify non-authoritative refinements. Classification
does not decide which source is authoritative and does not supersede either Memory.

<pairs>
{pairs_json}
</pairs>

Return a JSON object with a "decisions" array, one entry per pair:
{{"decisions": [{{"pair_index": 0, "classification": "contradiction", "reason": "Memory A says PostgreSQL 14, Memory B says MySQL"}}]}}

Return ONLY the JSON object."""


def _prompt_memory(memory: Memory) -> dict[str, str]:
    content = memory.content
    if len(content) > MAX_CONTRADICTION_MEMORY_CONTENT_CHARS:
        content = content[:MAX_CONTRADICTION_MEMORY_CONTENT_CHARS] + "\n[truncated]"
    return {"id": memory.id, "content": content, "type": memory.memory_type}


def _pairs_json(pairs: list[tuple[Memory, Memory]], *, pair_index_offset: int) -> str:
    return json.dumps(
        [
            {
                "pair_index": pair_index_offset + i,
                "memory_a": _prompt_memory(a),
                "memory_b": _prompt_memory(b),
            }
            for i, (a, b) in enumerate(pairs)
        ],
        indent=2,
    )


def _iter_prompt_sized_pair_batches(
    pairs: list[tuple[Memory, Memory]],
) -> list[tuple[int, list[tuple[Memory, Memory]]]]:
    batches: list[tuple[int, list[tuple[Memory, Memory]]]] = []
    start = 0
    batch_size = max(1, int(CONTRADICTION_LLM_BATCH_SIZE))
    while start < len(pairs):
        end = min(len(pairs), start + batch_size)
        while end > start:
            batch = pairs[start:end]
            prompt = CONTRADICTION_PROMPT.format(pairs_json=_pairs_json(batch, pair_index_offset=start))
            if len(prompt) <= MAX_CONTRADICTION_PROMPT_CHARS or len(batch) == 1:
                batches.append((start, batch))
                break
            end = start + max(1, len(batch) // 2)
        start = end
    return batches


def _validated_batch_decisions(
    decisions: list,
    *,
    batch_start: int,
    batch_size: int,
) -> list[dict]:
    serialized = [decision.model_dump() for decision in decisions]
    expected_indices = set(range(batch_start, batch_start + batch_size))
    actual_indices = [int(decision["pair_index"]) for decision in serialized]
    index_counts = Counter(actual_indices)
    duplicate_indices = {index for index, count in index_counts.items() if count > 1}
    actual_index_set = set(actual_indices)
    missing_indices = expected_indices - actual_index_set
    unexpected_indices = actual_index_set - expected_indices
    if len(serialized) != batch_size or duplicate_indices or missing_indices or unexpected_indices:
        raise StructuredLlmError(
            "contradiction decision coverage invalid: "
            f"expected_count={batch_size}, "
            f"actual_count={len(serialized)}, "
            f"missing_count={len(missing_indices)}, "
            f"duplicate_count={len(duplicate_indices)}, "
            f"unexpected_count={len(unexpected_indices)}"
        )
    return serialized


async def detect_cross_doc_contradictions(
    new_memory_ids: list[str],
    doc_id: str,
    db: Database,
    memory_store: MemoryStore,
    candidate_retriever: CrossDocumentCandidateRetriever,
    structured_llm_client=None,
    llm_model: str = "claude-sonnet-4-20250514",
    audit_context=None,
    actor_user_id: str | None = None,
) -> dict:
    """Classify cross-document relations for newly inserted Memories.

    Finds access-compatible Memories sharing entities with the new ones but
    from different source documents, then asks the LLM to classify each pair.

    Returns stats: {"contradictions": N, "temporal": N, "checked": N}
    """
    stats = {"contradictions": 0, "temporal": 0, "checked": 0, "truncated": 0}

    if not new_memory_ids or not structured_llm_client:
        return stats

    # Retrieve lightweight IDs/provenance first. Full Memory rows are loaded in
    # one bounded read only after hybrid fusion and exact access/source checks.
    pairs: list[tuple[Memory, Memory]] = []
    candidate_selection_by_challenger: dict[str, CrossDocumentCandidateSelection] = {}
    challenger_by_id: dict[str, Memory] = {}
    retrieval_selections: list[CrossDocumentCandidateSelection] = []
    evidence_unit_by_challenger: dict[str, EvidenceUnit] = {}
    had_preflight_failure = False
    llm_call_count = 0
    for mem_id in new_memory_ids:
        memory = await db.get_memory(mem_id)
        if not memory or memory.status != "active":
            continue

        entity_ids = await db.get_memory_entity_ids(mem_id)

        excluded_source_ids = await _disabled_source_ids_for_candidate_scope(
            db=db,
            challenger=memory,
            actor_user_id=actor_user_id,
        )
        selection = await candidate_retriever.retrieve(
            challenger=memory,
            entity_ids=entity_ids,
            doc_id=doc_id,
            actor_user_id=actor_user_id,
            excluded_source_ids=excluded_source_ids,
        )
        selection, loaded_by_id = await candidate_retriever.load_selected_memories(
            selection,
            challenger=memory,
            doc_id=doc_id,
            excluded_source_ids=excluded_source_ids,
        )
        retrieval_selections.append(selection)
        if not selection.candidate_ids:
            continue
        try:
            unit = await _cross_doc_evidence_unit(db=db, challenger=memory, doc_id=doc_id)
        except ValueError as error:
            had_preflight_failure = True
            await _record_detection_failed(
                memory_store=memory_store,
                audit_context=audit_context,
                doc_id=doc_id,
                llm_model=llm_model,
                new_memory_count=len(new_memory_ids),
                candidate_pairs=len(selection.candidate_ids),
                checked=0,
                llm_calls=0,
                retrieval_telemetry=_summarize_retrieval(retrieval_selections),
                error=str(error),
                reason="evidence_preflight_failure",
            )
            continue
        candidate_selection_by_challenger[mem_id] = selection
        challenger_by_id[mem_id] = memory
        evidence_unit_by_challenger[mem_id] = unit
        pairs.extend(
            (memory, loaded_by_id[candidate_id])
            for candidate_id in selection.candidate_ids
        )

    if not pairs:
        if had_preflight_failure:
            return stats
        await _record_detection_completed(
            memory_store=memory_store,
            audit_context=audit_context,
            doc_id=doc_id,
            llm_model=llm_model,
            new_memory_count=len(new_memory_ids),
            candidate_pairs=0,
            stats=stats,
            classifications={},
            llm_calls=0,
            retrieval_telemetry=_summarize_retrieval(retrieval_selections),
            reason="no_cross_doc_candidates",
        )
        return stats

    stats["checked"] = len(pairs)

    try:
        decisions = []
        for batch_start, batch in _iter_prompt_sized_pair_batches(pairs):
            pairs_json = _pairs_json(batch, pair_index_offset=batch_start)
            prompt = CONTRADICTION_PROMPT.format(pairs_json=pairs_json)
            llm_call_count += 1
            response = await structured_llm_client.detect_contradictions(
                prompt,
                max_tokens=DEFAULT_ENRICHMENT_MAX_TOKENS,
                model=llm_model,
            )
            decisions.extend(
                _validated_batch_decisions(
                    response.decisions,
                    batch_start=batch_start,
                    batch_size=len(batch),
                )
            )
        classifications = {
            "contradiction": 0,
            "temporal": 0,
            "clarification": 0,
            "unrelated": 0,
            "invalid": 0,
        }

        decisions_by_pair: dict[int, dict] = {}
        review_targets_by_challenger: dict[
            str,
            list[tuple[Memory, Memory, str, str]],
        ] = defaultdict(list)
        contradictions_to_record: list[tuple[str, str, str, str]] = []
        temporal_to_record: list[tuple[str, str, str, str]] = []
        for dec in decisions:
            idx = dec.get("pair_index", -1)
            if idx < 0 or idx >= len(pairs):
                classifications["invalid"] += 1
                continue

            classification = dec.get("classification", "unrelated").lower()
            classifications[classification if classification in classifications else "invalid"] += 1
            reason = dec.get("reason", "")
            mem_a, mem_b = pairs[idx]
            decisions_by_pair[idx] = {
                "classification": classification,
                "reason": reason,
            }

            if classification == "contradiction":
                review_targets_by_challenger[mem_a.id].append(
                    (mem_a, mem_b, reason, classification)
                )
                contradictions_to_record.append((mem_a.id, mem_b.id, "contradiction", reason))
                logger.info(
                    "CONTRADICTION: %s vs %s — %s",
                    mem_a.id,
                    mem_b.id,
                    reason,
                )

            elif classification == "temporal":
                temporal_to_record.append((mem_a.id, mem_b.id, "temporal", reason))
                logger.info(
                    "TEMPORAL: %s vs %s — %s",
                    mem_a.id,
                    mem_b.id,
                    reason,
                )

        for challenger_id, selection in candidate_selection_by_challenger.items():
            challenger = challenger_by_id[challenger_id]
            excluded_source_ids = await _disabled_source_ids_for_candidate_scope(
                db=db,
                challenger=challenger,
                actor_user_id=actor_user_id,
            )
            await candidate_retriever.ensure_selection_current(
                selection,
                challenger=challenger,
                doc_id=doc_id,
                excluded_source_ids=excluded_source_ids,
            )

        relation_outcomes = await _build_cross_doc_relation_outcome_bundles(
            pairs=pairs,
            decisions_by_pair=decisions_by_pair,
            doc_id=doc_id,
            db=db,
            candidate_selection_by_challenger=candidate_selection_by_challenger,
            evidence_unit_by_challenger=evidence_unit_by_challenger,
        )
        for challenger_id, bundle in relation_outcomes.items():
            review_targets = review_targets_by_challenger.get(challenger_id, ())
            if (
                review_targets
                and bundle.relation_run.lifecycle_action
                is LifecycleAction.CREATE_REVIEW
                and bundle.relation_run.review_case
                is ReviewCase.CROSS_SOURCE_CONFLICT
            ):
                for challenger, incumbent, reason, classification in review_targets:
                    await _record_cross_source_review(
                        challenger=challenger,
                        incumbent=incumbent,
                        reason=reason,
                        classification=classification,
                        db=db,
                        relation_outcome=bundle,
                    )
            else:
                await db.record_relation_outcome_bundle(bundle)
            for record in [
                item
                for item in contradictions_to_record
                if item[0] == challenger_id
            ]:
                await db.record_contradiction(*record)
                stats["contradictions"] += 1
        for record in temporal_to_record:
            await db.record_contradiction(*record)
            stats["temporal"] += 1

        await _record_detection_completed(
            memory_store=memory_store,
            audit_context=audit_context,
            doc_id=doc_id,
            llm_model=llm_model,
            new_memory_count=len(new_memory_ids),
            candidate_pairs=len(pairs),
            stats=stats,
            classifications=classifications,
            llm_calls=llm_call_count,
            retrieval_telemetry=_summarize_retrieval(retrieval_selections),
        )

    except StaleCandidateSelectionError as e:
        logger.info("Cross-document candidate selection changed before write: %s", e)
        await _record_detection_failed(
            memory_store=memory_store,
            audit_context=audit_context,
            doc_id=doc_id,
            llm_model=llm_model,
            new_memory_count=len(new_memory_ids),
            candidate_pairs=len(pairs),
            checked=stats["checked"],
            llm_calls=llm_call_count,
            retrieval_telemetry=_summarize_retrieval(retrieval_selections),
            error=str(e),
            reason="candidate_selection_stale",
        )
    except (StructuredLlmError, KeyError) as e:
        logger.warning("Structured contradiction detection failed: %s", e)
        await _record_detection_failed(
            memory_store=memory_store,
            audit_context=audit_context,
            doc_id=doc_id,
            llm_model=llm_model,
            new_memory_count=len(new_memory_ids),
            candidate_pairs=len(pairs),
            checked=stats["checked"],
            llm_calls=llm_call_count,
            retrieval_telemetry=_summarize_retrieval(retrieval_selections),
            error=str(e),
            reason="structured_output_failure",
        )
    except Exception as e:
        logger.error("Contradiction detection failed: %s", e)
        await _record_detection_failed(
            memory_store=memory_store,
            audit_context=audit_context,
            doc_id=doc_id,
            llm_model=llm_model,
            new_memory_count=len(new_memory_ids),
            candidate_pairs=len(pairs),
            checked=stats["checked"],
            llm_calls=llm_call_count,
            retrieval_telemetry=_summarize_retrieval(retrieval_selections),
            error=str(e),
            reason="runtime_failure",
        )

    if stats["contradictions"] > 0 or stats["temporal"] > 0:
        logger.info(
            "Cross-doc check: %d pairs checked, %d contradictions, %d temporal",
            stats["checked"],
            stats["contradictions"],
            stats["temporal"],
        )

    return stats


async def _disabled_source_ids_for_candidate_scope(
    *,
    db: Database,
    challenger: Memory,
    actor_user_id: str | None,
) -> list[str]:
    scope_user_id = (
        challenger.owner_user_id
        if challenger.visibility == "private"
        else actor_user_id
    )
    if not scope_user_id:
        return []
    return await db.list_disabled_source_ids_for_user(scope_user_id)


def _summarize_retrieval(
    selections: list[CrossDocumentCandidateSelection],
) -> dict[str, object]:
    latencies = [
        float(selection.telemetry.get("retrieval_latency_ms", 0.0))
        for selection in selections
    ]
    channel_candidate_counts: Counter[str] = Counter()
    for selection in selections:
        channel_candidate_counts.update(
            {
                str(channel): int(count)
                for channel, count in dict(
                    selection.telemetry.get("channel_candidate_counts", {})
                ).items()
            }
        )
    return {
        "challenger_count": len(selections),
        "rank_window_size": max(
            (
                int(selection.audit.get("rank_window_size", 0))
                for selection in selections
            ),
            default=0,
        ),
        "channel_candidate_counts": dict(sorted(channel_candidate_counts.items())),
        "eligible_candidate_count": sum(
            int(selection.telemetry.get("eligible_candidate_count", 0))
            for selection in selections
        ),
        "fused_candidate_count": sum(
            int(selection.telemetry.get("fused_candidate_count", 0))
            for selection in selections
        ),
        "selected_candidate_count": sum(len(selection.discovery) for selection in selections),
        "retrieval_latency_ms_total": round(sum(latencies), 3),
        "retrieval_latency_ms_max": round(max(latencies, default=0.0), 3),
        "provenance_rows_loaded": sum(
            int(selection.telemetry.get("provenance_rows_loaded", 0))
            for selection in selections
        ),
        "full_memory_rows_loaded": sum(
            int(selection.telemetry.get("full_memory_rows_loaded", 0))
            for selection in selections
        ),
        "channel_errors": sorted(
            {
                str(channel)
                for selection in selections
                for channel in selection.telemetry.get("channel_errors", ())
            }
        ),
    }


async def _record_detection_completed(
    *,
    memory_store: MemoryStore,
    audit_context,
    doc_id: str,
    llm_model: str,
    new_memory_count: int,
    candidate_pairs: int,
    stats: dict,
    classifications: dict[str, int],
    llm_calls: int,
    retrieval_telemetry: dict[str, object],
    reason: str | None = None,
) -> None:
    if not hasattr(memory_store, "record_audit_event"):
        return
    context = audit_context
    if context is None and hasattr(memory_store, "operation_context"):
        context = memory_store.operation_context(doc_id=doc_id)
    await memory_store.record_audit_event(
        "contradiction_detection_completed",
        "committed",
        context=context,
        doc_id=doc_id,
        model=llm_model,
        reason=reason,
        payload={
            "new_memory_count": new_memory_count,
            "candidate_pairs": candidate_pairs,
            "checked": stats["checked"],
            "contradictions": stats["contradictions"],
            "temporal": stats["temporal"],
            "truncated": stats.get("truncated", 0),
            "classifications": classifications,
            "llm_calls": llm_calls,
            "retrieval": retrieval_telemetry,
        },
    )


async def _record_detection_failed(
    *,
    memory_store: MemoryStore,
    audit_context,
    doc_id: str,
    llm_model: str,
    new_memory_count: int,
    candidate_pairs: int,
    checked: int,
    llm_calls: int,
    retrieval_telemetry: dict[str, object],
    error: str,
    reason: str,
) -> None:
    if not hasattr(memory_store, "record_audit_event"):
        return
    context = audit_context
    if context is None and hasattr(memory_store, "operation_context"):
        context = memory_store.operation_context(doc_id=doc_id)
    await memory_store.record_audit_event(
        "contradiction_detection_failed",
        "failed",
        context=context,
        doc_id=doc_id,
        model=llm_model,
        reason=reason,
        error=error,
        payload={
            "new_memory_count": new_memory_count,
            "candidate_pairs": candidate_pairs,
            "checked": checked,
            "llm_calls": llm_calls,
            "retrieval": retrieval_telemetry,
            "truncated": 0,
        },
    )


async def _build_cross_doc_relation_outcome_bundles(
    *,
    pairs: list[tuple[Memory, Memory]],
    decisions_by_pair: dict[int, dict],
    doc_id: str,
    db: Database,
    candidate_selection_by_challenger: dict[str, CrossDocumentCandidateSelection],
    evidence_unit_by_challenger: dict[str, EvidenceUnit] | None = None,
) -> dict[str, RelationOutcomeBundle]:
    bundles: dict[str, RelationOutcomeBundle] = {}
    pairs_by_challenger: dict[str, list[tuple[int, Memory, Memory]]] = defaultdict(list)
    for index, (challenger, incumbent) in enumerate(pairs):
        pairs_by_challenger[challenger.id].append((index, challenger, incumbent))

    for challenger_id, challenger_pairs in pairs_by_challenger.items():
        relation_records: list[EvidenceRelationRecord] = []

        _, challenger, _ = challenger_pairs[0]
        selection = candidate_selection_by_challenger[challenger.id]
        unit = (evidence_unit_by_challenger or {}).get(challenger.id)
        if unit is None:
            unit = await _cross_doc_evidence_unit(db=db, challenger=challenger, doc_id=doc_id)
        classified_pairs = [
            (
                pair_index,
                incumbent,
                str(decisions_by_pair.get(pair_index, {}).get("classification") or "unrelated"),
                str(decisions_by_pair.get(pair_index, {}).get("reason") or ""),
            )
            for pair_index, _, incumbent in challenger_pairs
        ]
        has_contradiction = any(
            classification == "contradiction"
            for _, _, classification, _ in classified_pairs
        )
        has_refinement = any(
            classification in {"temporal", "clarification"}
            for _, _, classification, _ in classified_pairs
        )
        if has_contradiction:
            lifecycle_action = LifecycleAction.CREATE_REVIEW
            review_case = ReviewCase.CROSS_SOURCE_CONFLICT
        else:
            lifecycle_action = LifecycleAction.NONE
            review_case = None
        identity_relation_type = None
        identity_authority_case = None
        if has_contradiction:
            identity_relation_type = RelationType.CONTRADICTS
            identity_authority_case = AuthorityCase.CROSS_SOURCE_CONFLICT
        elif has_refinement:
            identity_relation_type = RelationType.REFINES
            identity_authority_case = AuthorityCase.INDEPENDENT_REFINEMENT
        relation_run_id = _cross_doc_relation_run_id(
            unit,
            action=lifecycle_action,
            relation_type=identity_relation_type,
            authority_case=identity_authority_case,
            candidate_snapshot_identity=selection.snapshot_identity,
        )
        universe = build_candidate_universe(
            relation_run_id=relation_run_id,
            evidence_unit_id=unit.id,
            bucket_results=selection.bucket_results(),
            recall_candidate_cap=max(1, len(selection.discovery)),
        )
        candidate_records = list(universe.candidates)
        candidate_record_by_id = {
            candidate.memory_id: candidate for candidate in candidate_records
        }

        for _, incumbent, classification, reason in classified_pairs:
            if classification not in {"contradiction", "temporal", "clarification"}:
                continue
            if incumbent.id not in candidate_record_by_id:
                raise ValueError("classified candidate is absent from the candidate universe")
            relation_type = RelationType.CONTRADICTS if classification == "contradiction" else RelationType.REFINES
            authority_case = (
                AuthorityCase.CROSS_SOURCE_CONFLICT
                if classification == "contradiction"
                else AuthorityCase.INDEPENDENT_REFINEMENT
            )
            relation_records.append(
                EvidenceRelationRecord(
                    evidence_unit_id=unit.id,
                    memory_id=incumbent.id,
                    relation_type=relation_type,
                    authority_case=authority_case,
                    is_authoritative_support=False,
                    source_lineage_id=unit.source_lineage_id,
                    confidence=1.0,
                    reason=reason,
                    excerpt=None,
                    classifier_version="cross-doc-contradiction-v2",
                    relation_run_id=relation_run_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )

        relation_run = RelationRunRecord(
            id=relation_run_id,
            evidence_unit_id=unit.id,
            access_context_hash=unit.access_context_hash,
            candidate_count=len(candidate_records),
            mandatory_candidate_count=universe.mandatory_candidate_count,
            checked_candidate_count=universe.checked_candidate_count,
            incomplete_mandatory_buckets=universe.incomplete_mandatory_buckets,
            classifier_version="cross-doc-contradiction-v2",
            lifecycle_action=lifecycle_action,
            review_case=review_case,
            status=(
                "review"
                if lifecycle_action is LifecycleAction.CREATE_REVIEW
                else "checked"
            ),
            audit={
                "source": "detect_cross_doc_contradictions",
                "challenger_memory_id": challenger_id,
                "candidate_pair_count": len(candidate_records),
                **selection.audit,
            },
        )
        bundles[challenger_id] = RelationOutcomeBundle(
            evidence_unit=unit,
            relation_run=relation_run,
            candidates=tuple(candidate_records),
            relations=tuple(relation_records),
        )
    return bundles


async def _cross_doc_evidence_unit(*, db: Database, challenger: Memory, doc_id: str) -> EvidenceUnit:
    document = await db.get_document(doc_id)
    if document is None or not document.source:
        raise ValueError("cross-document relation requires an authoritative source document")
    support = await db.get_active_memory_support_evidence(
        challenger.id,
        source_id=document.source,
    )
    primary_support_by_unit: dict[str, list] = defaultdict(list)
    for item in support:
        if item.role is EvidenceRole.PRIMARY:
            primary_support_by_unit[item.evidence_unit_id].append(item)
    units: list[EvidenceUnit] = []
    for evidence_unit_id, primary_support in primary_support_by_unit.items():
        unit = await db.get_evidence_unit(evidence_unit_id)
        if (
            unit is None
            or unit.source_id != document.source
            or unit.doc_id != doc_id
            or not unit.access_context_hash
            or not unit.source_lineage_id
        ):
            continue
        current_revisions = await db.get_current_source_observation_revisions(
            unit.source_lineage_id
        )
        if any(
            (
                current := current_revisions.get(item.anchor.observation_id)
            ) is not None
            and current.id == item.anchor.observation_revision_id
            for item in primary_support
        ):
            units.append(unit)
    if len(units) != 1:
        raise ValueError(
            "cross-document relation requires exactly one current primary Support Evidence Unit"
        )
    return units[0]


def _cross_doc_relation_run_id(
    unit: EvidenceUnit,
    *,
    action: LifecycleAction = LifecycleAction.CREATE_REVIEW,
    relation_type: RelationType | None = RelationType.CONTRADICTS,
    authority_case: AuthorityCase | None = AuthorityCase.CROSS_SOURCE_CONFLICT,
    candidate_snapshot_identity: str | None = None,
) -> str:
    return relation_run_id_for(
        prefix="contradiction",
        unit=unit,
        action=action,
        classifier_version="cross-doc-contradiction-v2",
        candidate_memory_id=(
            f"candidate-snapshot:{candidate_snapshot_identity}"
            if candidate_snapshot_identity
            else None
        ),
        relation_type=relation_type,
        authority_case=authority_case,
        bucket=CandidateBucket.HYBRID_DISCOVERY,
    )


async def _record_cross_source_review(
    *,
    challenger: Memory,
    incumbent: Memory,
    reason: str | None,
    classification: str,
    db: Database,
    relation_outcome: RelationOutcomeBundle | None = None,
) -> None:
    """Record a non-destructive cross-source review without changing Memory status."""
    assert relation_outcome is not None
    latest_challenger = await db.get_memory(challenger.id)
    review = MemoryReview(
        id=generate_deterministic_review_id(
            kind=ReviewKind.CROSS_SOURCE_CONFLICT.value,
            incumbent_memory_id=incumbent.id,
            challenger_memory_id=challenger.id,
            relation_run_id=relation_outcome.relation_run.id if relation_outcome else None,
            evidence_unit_id=relation_outcome.evidence_unit.id if relation_outcome else None,
            review_case=(
                relation_outcome.relation_run.review_case.value
                if relation_outcome and relation_outcome.relation_run.review_case
                else None
            ),
        ),
        kind=ReviewKind.CROSS_SOURCE_CONFLICT.value,
        status=ReviewStatus.PENDING.value,
        incumbent_memory_id=incumbent.id,
        challenger_memory_id=challenger.id,
        reason=f"{classification}: {reason}" if reason else classification,
        expected_incumbent_updated_at=(incumbent.updated_at.isoformat() if incumbent.updated_at else None),
        expected_challenger_updated_at=(
            latest_challenger.updated_at.isoformat() if latest_challenger and latest_challenger.updated_at else None
        ),
        created_at=datetime.now(timezone.utc),
    )
    await db.record_memory_review_with_relation_outcome(
        review,
        relation_outcome,
    )
