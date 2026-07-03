"""Cross-document contradiction detection.

After memories are inserted from a document sync, checks whether any newly
inserted memories contradict existing memories from OTHER documents that
reference the same entities. Uses entity overlap as the candidate signal
and LLM classification for the actual judgment.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from typing import TYPE_CHECKING

from memforge.config import DEFAULT_ENRICHMENT_MAX_TOKENS
from memforge.llm.structured import StructuredLlmError
from memforge.memory.evidence import (
    AuthorityCase,
    CandidateBucket,
    EvidenceContentProvenance,
    EvidenceRelationRecord,
    EvidenceUnit,
    LifecycleAction,
    MemoryRelationApplyService,
    RelationCandidateRecord,
    RelationDecision,
    RelationOutcomeBundle,
    RelationRunRecord,
    RelationType,
    ReviewCase,
    relation_run_id_for,
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

MAX_CONTRADICTION_PAIRS_PER_RUN = 200
CONTRADICTION_LLM_BATCH_SIZE = 20
MAX_CONTRADICTION_PROMPT_CHARS = 120_000
MAX_CONTRADICTION_MEMORY_CONTENT_CHARS = 4_000


CONTRADICTION_PROMPT = """You are checking whether pairs of team knowledge memories contradict each other.
These memories come from different source documents about the same entities.

For each pair, classify the relationship:
- CONTRADICTION: they make mutually incompatible claims about the same thing
- TEMPORAL: they describe the same fact at different time points (the newer one supersedes)
- CLARIFICATION: one adds detail to the other without conflicting
- UNRELATED: they happen to share an entity but discuss different aspects

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
            prompt = CONTRADICTION_PROMPT.format(
                pairs_json=_pairs_json(batch, pair_index_offset=start)
            )
            if len(prompt) <= MAX_CONTRADICTION_PROMPT_CHARS or len(batch) == 1:
                batches.append((start, batch))
                break
            end = start + max(1, len(batch) // 2)
        start = end
    return batches


async def detect_cross_doc_contradictions(
    new_memory_ids: list[str],
    doc_id: str,
    db: Database,
    memory_store: MemoryStore,
    structured_llm_client=None,
    llm_model: str = "claude-sonnet-4-20250514",
    audit_context=None,
    actor_user_id: str | None = None,
) -> dict:
    """Check newly inserted memories for contradictions with other documents.

    Finds existing memories sharing entities with the new ones but from
    different source documents, then asks the LLM to classify each pair.

    Returns stats: {"contradictions": N, "temporal": N, "checked": N}
    """
    stats = {"contradictions": 0, "temporal": 0, "checked": 0, "truncated": 0}

    if not new_memory_ids or not structured_llm_client:
        return stats

    # Collect candidate pairs: new memory + cross-doc memory sharing entities.
    # This is a memory boundary as well as a behavior boundary: large workspaces
    # can produce many candidates for a single synced document.
    pairs: list[tuple[Memory, Memory]] = []
    candidate_bucket_complete_by_challenger: dict[str, bool] = {}

    for mem_id in new_memory_ids:
        if len(pairs) >= MAX_CONTRADICTION_PAIRS_PER_RUN:
            memory = await db.get_memory(mem_id)
            if memory and memory.status == "active":
                await _record_truncated_cross_doc_candidate_page(
                    challenger=memory,
                    candidates=(),
                    doc_id=doc_id,
                    db=db,
                )
                stats["truncated"] += 1
            continue

        cap_reached_for_memory = False
        memory = await db.get_memory(mem_id)
        if not memory or memory.status != "active":
            continue

        entity_ids = await db.get_memory_entity_ids(mem_id)
        if not entity_ids:
            continue

        excluded_source_ids: list[str] = []
        scope_user_id = memory.owner_user_id if memory.visibility == "private" else None
        if scope_user_id:
            excluded_source_ids = await db.list_disabled_source_ids_for_user(scope_user_id)
        candidate_page = await db.get_cross_doc_candidates(
            mem_id,
            entity_ids,
            doc_id,
            owner_user_id=memory.owner_user_id,
            visibility=memory.visibility,
            project_key=memory.project_key,
            excluded_source_ids=excluded_source_ids,
        )
        candidate_bucket_complete_by_challenger[mem_id] = candidate_page.complete
        if not candidate_page.complete:
            await _record_truncated_cross_doc_candidate_page(
                challenger=memory,
                candidates=tuple(candidate_page.candidates),
                doc_id=doc_id,
                db=db,
            )
            stats["truncated"] += 1
            continue
        for candidate in candidate_page.candidates:
            if len(pairs) >= MAX_CONTRADICTION_PAIRS_PER_RUN:
                cap_reached_for_memory = True
                candidate_bucket_complete_by_challenger[mem_id] = False
                break
            pairs.append((memory, candidate))
        if cap_reached_for_memory:
            stats["truncated"] += 1
            continue

    if not pairs:
        await _record_detection_completed(
            memory_store=memory_store,
            audit_context=audit_context,
            doc_id=doc_id,
            llm_model=llm_model,
            new_memory_count=len(new_memory_ids),
            candidate_pairs=0,
            stats=stats,
            classifications={},
            reason="no_cross_doc_candidates",
        )
        return stats

    stats["checked"] = len(pairs)

    try:
        decisions = []
        for batch_start, batch in _iter_prompt_sized_pair_batches(pairs):
            pairs_json = _pairs_json(batch, pair_index_offset=batch_start)
            prompt = CONTRADICTION_PROMPT.format(pairs_json=pairs_json)
            response = await structured_llm_client.detect_contradictions(
                prompt,
                max_tokens=DEFAULT_ENRICHMENT_MAX_TOKENS,
                model=llm_model,
            )
            decisions.extend(decision.model_dump() for decision in response.decisions)
        classifications = {
            "contradiction": 0,
            "temporal": 0,
            "clarification": 0,
            "unrelated": 0,
            "invalid": 0,
        }

        decisions_by_pair: dict[int, dict] = {}
        challengers_for_review: dict[str, tuple[Memory, Memory, str]] = {}
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
                challengers_for_review.setdefault(mem_a.id, (mem_a, mem_b, reason))
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

        relation_outcomes = await _build_cross_doc_relation_outcome_bundles(
            pairs=pairs,
            decisions_by_pair=decisions_by_pair,
            doc_id=doc_id,
            db=db,
            candidate_bucket_complete_by_challenger=candidate_bucket_complete_by_challenger,
        )
        for challenger_id, bundle in relation_outcomes.items():
            review_target = challengers_for_review.get(challenger_id)
            if review_target is not None and bundle.relation_run.lifecycle_action is LifecycleAction.CREATE_REVIEW:
                challenger, incumbent, reason = review_target
                await _quarantine_challenger(
                    challenger=challenger,
                    incumbent=incumbent,
                    reason=reason,
                    db=db,
                    memory_store=memory_store,
                    relation_outcome=bundle,
                )
                for record in [item for item in contradictions_to_record if item[0] == challenger_id]:
                    await db.record_contradiction(*record)
                    stats["contradictions"] += 1
            else:
                await db.record_relation_outcome_bundle(bundle)
                for record in [item for item in contradictions_to_record if item[0] == challenger_id]:
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
    reason: str | None = None,
) -> None:
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
    error: str,
    reason: str,
) -> None:
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
            "truncated": 0,
        },
    )


async def _build_cross_doc_relation_outcome_bundles(
    *,
    pairs: list[tuple[Memory, Memory]],
    decisions_by_pair: dict[int, dict],
    doc_id: str,
    db: Database,
    candidate_bucket_complete_by_challenger: dict[str, bool],
) -> dict[str, RelationOutcomeBundle]:
    bundles: dict[str, RelationOutcomeBundle] = {}
    pairs_by_challenger: dict[str, list[tuple[int, Memory, Memory]]] = defaultdict(list)
    for index, (challenger, incumbent) in enumerate(pairs):
        pairs_by_challenger[challenger.id].append((index, challenger, incumbent))

    for challenger_id, challenger_pairs in pairs_by_challenger.items():
        decisions: list[RelationDecision] = []
        relation_records: list[EvidenceRelationRecord] = []
        candidate_records: list[RelationCandidateRecord] = []

        _, challenger, _ = challenger_pairs[0]
        bucket_complete = candidate_bucket_complete_by_challenger.get(challenger.id, True)
        unit = await _cross_doc_evidence_unit(db=db, challenger=challenger, doc_id=doc_id)
        relation_run_id = _cross_doc_relation_run_id(unit)

        for candidate_rank, (pair_index, _, incumbent) in enumerate(challenger_pairs):
            classification = str(decisions_by_pair.get(pair_index, {}).get("classification") or "unrelated")
            reason = str(decisions_by_pair.get(pair_index, {}).get("reason") or "")
            candidate_records.append(
                RelationCandidateRecord(
                    relation_run_id=relation_run_id,
                    evidence_unit_id=unit.id,
                    memory_id=incumbent.id,
                    bucket=CandidateBucket.SHARED_ENTITIES,
                    bucket_rank=0,
                    candidate_rank=candidate_rank,
                    score=None,
                    is_mandatory=False,
                    bucket_complete=bucket_complete,
                    was_checked=True,
                    reason="cross_doc_entity_overlap",
                )
            )
            if classification != "contradiction":
                continue
            decision = RelationDecision(
                candidate_memory_id=incumbent.id,
                relation_type=RelationType.CONTRADICTS,
                authority_case=AuthorityCase.CROSS_SOURCE_CONFLICT,
                confidence=1.0,
                reason=reason,
                evidence_excerpt=None,
                matched_bucket=CandidateBucket.SHARED_ENTITIES,
                matched_bucket_complete=bucket_complete,
                classifier_batch_key=relation_run_id,
            )
            decisions.append(decision)
            relation_records.append(
                EvidenceRelationRecord(
                    evidence_unit_id=unit.id,
                    memory_id=incumbent.id,
                    relation_type=RelationType.CONTRADICTS,
                    authority_case=AuthorityCase.CROSS_SOURCE_CONFLICT,
                    is_authoritative_support=False,
                    source_lineage_id=unit.source_lineage_id,
                    confidence=1.0,
                    reason=reason,
                    excerpt=None,
                    classifier_version="cross-doc-contradiction-v1",
                    relation_run_id=relation_run_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )

        if decisions:
            lifecycle = MemoryRelationApplyService().derive_lifecycle(unit, decisions)
            lifecycle_action = lifecycle.action
            review_case = lifecycle.review_case
        elif not bucket_complete:
            lifecycle_action = LifecycleAction.CREATE_REVIEW
            review_case = ReviewCase.MANDATORY_INCOMPLETE
        else:
            lifecycle_action = LifecycleAction.NONE
            review_case = None
        relation_run = RelationRunRecord(
            id=relation_run_id,
            evidence_unit_id=unit.id,
            access_context_hash=unit.access_context_hash,
            candidate_count=len(candidate_records),
            mandatory_candidate_count=sum(1 for candidate in candidate_records if candidate.is_mandatory),
            checked_candidate_count=sum(1 for candidate in candidate_records if candidate.was_checked),
            incomplete_mandatory_buckets=(),
            classifier_version="cross-doc-contradiction-v1",
            lifecycle_action=lifecycle_action,
            review_case=review_case,
            status=(
                "checked"
                if lifecycle_action is LifecycleAction.NONE
                else "review_required"
                if review_case is ReviewCase.MANDATORY_INCOMPLETE
                else "review"
                if lifecycle_action is LifecycleAction.CREATE_REVIEW
                else "applied"
            ),
            audit={
                "source": "detect_cross_doc_contradictions",
                "challenger_memory_id": challenger_id,
                "candidate_pair_count": len(candidate_records),
                "candidate_bucket_complete": bucket_complete,
            },
        )
        bundles[challenger_id] = RelationOutcomeBundle(
            evidence_unit=unit,
            relation_run=relation_run,
            candidates=tuple(candidate_records),
            relations=tuple(relation_records),
        )
    return bundles


async def _record_truncated_cross_doc_candidate_page(
    *,
    challenger: Memory,
    candidates: tuple[Memory, ...],
    doc_id: str,
    db: Database,
) -> None:
    unit = await _cross_doc_evidence_unit(db=db, challenger=challenger, doc_id=doc_id)
    relation_run_id = _cross_doc_relation_run_id(unit)
    candidate_records = tuple(
        RelationCandidateRecord(
            relation_run_id=relation_run_id,
            evidence_unit_id=unit.id,
            memory_id=candidate.id,
            bucket=CandidateBucket.SHARED_ENTITIES,
            bucket_rank=0,
            candidate_rank=index,
            score=None,
            is_mandatory=False,
            bucket_complete=False,
            was_checked=False,
            reason="cross_doc_candidate_page_incomplete",
        )
        for index, candidate in enumerate(candidates)
    )
    await db.record_relation_outcome_bundle(
        RelationOutcomeBundle(
            evidence_unit=unit,
            relation_run=RelationRunRecord(
                id=relation_run_id,
                evidence_unit_id=unit.id,
                access_context_hash=unit.access_context_hash,
                candidate_count=len(candidate_records),
                mandatory_candidate_count=0,
                checked_candidate_count=0,
                incomplete_mandatory_buckets=(CandidateBucket.SHARED_ENTITIES.value,),
                classifier_version="cross-doc-contradiction-v1",
                lifecycle_action=LifecycleAction.CREATE_REVIEW,
                review_case=ReviewCase.MANDATORY_INCOMPLETE,
                status="review_required",
                audit={
                    "source": "detect_cross_doc_contradictions",
                    "challenger_memory_id": challenger.id,
                    "candidate_bucket_complete": False,
                    "reason": "candidate_page_incomplete",
                },
            ),
            candidates=candidate_records,
            relations=(),
        )
    )


async def _cross_doc_evidence_unit(*, db: Database, challenger: Memory, doc_id: str) -> EvidenceUnit:
    sources = await db.get_memory_sources(challenger.id)
    source = next((item for item in sources if item.doc_id == doc_id), sources[0] if sources else None)
    document = await db.get_document(doc_id)
    source_id = document.source if document is not None and document.source else "unknown"
    source_type = source.source_type if source is not None else "unknown"
    doc_revision_id = None
    if document is not None:
        doc_revision_id = document.content_hash or document.version
    unit_id = _cross_doc_evidence_unit_id(
        source_id=source_id,
        doc_id=doc_id,
        doc_revision_id=doc_revision_id,
        challenger_memory_id=challenger.id,
    )
    return EvidenceUnit(
        id=unit_id,
        source_id=source_id,
        doc_id=doc_id,
        doc_revision_id=doc_revision_id,
        source_type=source_type,
        source_anchor=challenger.id,
        source_lineage_id=doc_id,
        project_key=challenger.project_key,
        visibility=challenger.visibility,
        owner_user_id=challenger.owner_user_id,
        repo_identifier=challenger.repo_identifier,
        content=challenger.content,
        excerpt=None,
        evidence_provenance=EvidenceContentProvenance.NO_EXCERPT,
        source_metadata={
            "challenger_memory_id": challenger.id,
            "detection": "cross_doc_contradiction",
        },
        observed_at=datetime.now(timezone.utc).isoformat(),
    )


def _cross_doc_evidence_unit_id(
    *,
    source_id: str,
    doc_id: str,
    doc_revision_id: str | None,
    challenger_memory_id: str,
) -> str:
    digest = sha256("\x1f".join([source_id, doc_id, doc_revision_id or "", challenger_memory_id]).encode()).hexdigest()
    return f"eu-contradiction-{digest[:16]}"


def _cross_doc_relation_run_id(unit: EvidenceUnit, memory_id: str | None = None) -> str:
    return relation_run_id_for(
        prefix="contradiction",
        unit=unit,
        action=LifecycleAction.CREATE_REVIEW,
        classifier_version="cross-doc-contradiction-v1",
        candidate_memory_id=memory_id,
        relation_type=RelationType.CONTRADICTS,
        authority_case=AuthorityCase.CROSS_SOURCE_CONFLICT,
        bucket=CandidateBucket.SHARED_ENTITIES,
    )


async def _quarantine_challenger(
    *,
    challenger: Memory,
    incumbent: Memory,
    reason: str | None,
    db: Database,
    memory_store: MemoryStore,
    relation_outcome: RelationOutcomeBundle | None = None,
) -> None:
    """Hold a conflicting challenger for the existing review workbench."""
    existing = await db.get_pending_review_for_challenger(challenger.id)
    if existing:
        return

    for source in await db.get_memory_sources(challenger.id):
        existing_case = await db.get_open_review_for_incumbent_source_doc(
            incumbent_memory_id=incumbent.id,
            doc_id=source.doc_id,
            kind=ReviewKind.SUPERSEDE.value,
        )
        if existing_case:
            await memory_store.mark_pending_review_with_case(
                challenger.id,
                reason=reason,
                relation_outcome=relation_outcome,
                related_review_id=existing_case.id,
            )
            return

    latest_challenger = await db.get_memory(challenger.id)
    review = MemoryReview(
        id=generate_deterministic_review_id(
            kind=ReviewKind.SUPERSEDE.value,
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
        kind=ReviewKind.SUPERSEDE.value,
        status=ReviewStatus.PENDING.value,
        incumbent_memory_id=incumbent.id,
        challenger_memory_id=challenger.id,
        reason=reason,
        expected_incumbent_updated_at=(incumbent.updated_at.isoformat() if incumbent.updated_at else None),
        expected_challenger_updated_at=(
            latest_challenger.updated_at.isoformat() if latest_challenger and latest_challenger.updated_at else None
        ),
        created_at=datetime.now(timezone.utc),
    )
    await memory_store.mark_pending_review_with_case(
        challenger.id,
        reason=reason,
        relation_outcome=relation_outcome,
        review=review,
    )
