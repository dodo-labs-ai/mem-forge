"""Cross-document contradiction detection.

After memories are inserted from a document sync, checks whether any newly
inserted memories contradict existing memories from OTHER documents that
reference the same entities. Uses entity overlap as the candidate signal
and LLM classification for the actual judgment.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from meminception.llm.structured import StructuredLlmError
from meminception.models import (
    Memory,
    MemoryReview,
    ReviewKind,
    ReviewStatus,
    generate_review_id,
)

if TYPE_CHECKING:
    from meminception.memory.store import MemoryStore
    from meminception.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = ["detect_cross_doc_contradictions"]


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


async def detect_cross_doc_contradictions(
    new_memory_ids: list[str],
    doc_id: str,
    db: Database,
    memory_store: MemoryStore,
    structured_llm_client=None,
    llm_model: str = "claude-sonnet-4-20250514",
    audit_context=None,
) -> dict:
    """Check newly inserted memories for contradictions with other documents.

    Finds existing memories sharing entities with the new ones but from
    different source documents, then asks the LLM to classify each pair.

    Returns stats: {"contradictions": N, "temporal": N, "checked": N}
    """
    stats = {"contradictions": 0, "temporal": 0, "checked": 0}

    if not new_memory_ids or not structured_llm_client:
        return stats

    # Collect candidate pairs: new memory + cross-doc memory sharing entities
    pairs: list[tuple[Memory, Memory]] = []

    for mem_id in new_memory_ids:
        memory = await db.get_memory(mem_id)
        if not memory or memory.status != "active":
            continue

        entity_ids = await db.get_memory_entity_ids(mem_id)
        if not entity_ids:
            continue

        candidates = await db.get_cross_doc_candidates(mem_id, entity_ids, doc_id)
        for candidate in candidates:
            pairs.append((memory, candidate))

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

    # Batch LLM classification
    pairs_json = json.dumps([
        {
            "pair_index": i,
            "memory_a": {"id": a.id, "content": a.content, "type": a.memory_type},
            "memory_b": {"id": b.id, "content": b.content, "type": b.memory_type},
        }
        for i, (a, b) in enumerate(pairs)
    ], indent=2)

    prompt = CONTRADICTION_PROMPT.format(pairs_json=pairs_json)

    try:
        response = await structured_llm_client.detect_contradictions(
            prompt,
            max_tokens=64000,
            model=llm_model,
        )
        decisions = [decision.model_dump() for decision in response.decisions]
        classifications = {
            "contradiction": 0,
            "temporal": 0,
            "clarification": 0,
            "unrelated": 0,
            "invalid": 0,
        }

        for dec in decisions:
            idx = dec.get("pair_index", -1)
            if idx < 0 or idx >= len(pairs):
                classifications["invalid"] += 1
                continue

            classification = dec.get("classification", "unrelated").lower()
            classifications[classification if classification in classifications else "invalid"] += 1
            reason = dec.get("reason", "")
            mem_a, mem_b = pairs[idx]

            if classification == "contradiction":
                await db.record_contradiction(mem_a.id, mem_b.id, "contradiction", reason)
                await _quarantine_challenger(
                    challenger=mem_a,
                    incumbent=mem_b,
                    reason=reason,
                    db=db,
                    memory_store=memory_store,
                )
                stats["contradictions"] += 1
                logger.info(
                    "CONTRADICTION: %s vs %s — %s",
                    mem_a.id, mem_b.id, reason,
                )

            elif classification == "temporal":
                await db.record_contradiction(mem_a.id, mem_b.id, "temporal", reason)
                stats["temporal"] += 1
                logger.info(
                    "TEMPORAL: %s vs %s — %s",
                    mem_a.id, mem_b.id, reason,
                )

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
            stats["checked"], stats["contradictions"], stats["temporal"],
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
        },
    )


async def _quarantine_challenger(
    *,
    challenger: Memory,
    incumbent: Memory,
    reason: str | None,
    db: Database,
    memory_store: MemoryStore,
) -> None:
    """Hold a conflicting challenger for the existing review workbench."""
    await memory_store.mark_pending_review(challenger.id, reason=reason)

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
            await db.add_memory_review_related_challenger(
                existing_case.id,
                challenger.id,
                reason=reason,
            )
            return

    latest_challenger = await db.get_memory(challenger.id)
    review = MemoryReview(
        id=generate_review_id(),
        kind=ReviewKind.SUPERSEDE.value,
        status=ReviewStatus.PENDING.value,
        incumbent_memory_id=incumbent.id,
        challenger_memory_id=challenger.id,
        reason=reason,
        expected_incumbent_updated_at=(
            incumbent.updated_at.isoformat() if incumbent.updated_at else None
        ),
        expected_challenger_updated_at=(
            latest_challenger.updated_at.isoformat()
            if latest_challenger and latest_challenger.updated_at
            else None
        ),
        created_at=datetime.now(timezone.utc),
    )
    await db.insert_memory_review(review)
