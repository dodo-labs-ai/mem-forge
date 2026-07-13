"""Memory Engine — orchestrates entity resolution, dedup, and storage.

This is the bridge between the enrichment/extraction pipeline and the memory store.
It processes enrichment results (Call 1) and extracted memories (Call 2) into
persisted, deduplicated, entity-linked memories.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from memforge.memory.entity_resolver import EntityResolver, insert_llm_alias, resolve_entity
from memforge.memory.evidence import (
    AccessContext,
    AuthorityCase,
    EvidenceContentProvenance,
    EvidenceRelationRecord,
    EvidenceUnit,
    LifecycleAction,
    LifecycleDecision,
    MemoryRelationApplyService,
    RelationCandidateRecord,
    RelationDecision,
    RelationOutcomeBundle,
    RelationRunRecord,
    RelationType,
    ReviewCase,
    build_candidate_universe,
    build_mandatory_candidate_bucket_results,
    relation_run_id_for,
)
from memforge.memory.lifecycle import requires_human_review
from memforge.memory.quality import classify_memory_candidate
from memforge.source_access import memory_visibility_for_document
from memforge.models import (
    Memory,
    MemoryReview,
    RawMemory,
    ReconcileAction,
    ReconcileOperation,
    ReplacementKind,
    ReviewKind,
    ReviewStatus,
    content_hash,
    generate_memory_id,
    generate_deterministic_review_id,
)

from memforge.storage.adapters.protocols import RelationalStore, VectorStore

if TYPE_CHECKING:
    from memforge.memory.store import MemoryStore
    from memforge.models import EnrichmentResult
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = ["MemoryEngine"]


class MemoryEngine:
    """Orchestrates the flow from LLM extraction output to persisted memories.

    Responsibilities:
    - Process Call 1 output: resolve entities, insert aliases
    - Process Call 2 output: build Memory objects, deduplicate, insert/corroborate
    """

    def __init__(
        self,
        relational: RelationalStore,
        vector: VectorStore,
        db: Database,
        memory_store: MemoryStore,
        embed_cfg: dict | None = None,
        structured_llm_client: Any = None,
        llm_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        # Held so a later phase can stamp visibility through them without
        # re-plumbing this constructor; the orchestration here reads neither yet.
        self.relational = relational
        self.vector = vector
        self.db = db
        self.memory_store = memory_store
        self.structured_llm_client = structured_llm_client
        self.llm_model = llm_model
        # Entity resolver with embedding + LLM capabilities
        self.entity_resolver = EntityResolver(
            db=db,
            embed_cfg=embed_cfg,
            structured_llm_client=structured_llm_client,
            llm_model=llm_model,
        )

    # -------------------------------------------------------------------
    # Process Call 1: Entity resolution + alias insertion
    # -------------------------------------------------------------------

    async def process_enrichment(
        self,
        doc_id: str,
        enrichment: EnrichmentResult,
        doc_context: str | None = None,
    ) -> list[int]:
        """Process enrichment result: resolve entities and register their aliases.

        For each entity extracted by Call 1, resolves it against the DB
        (exact match → alias lookup → embedding search → create new),
        then registers any aliases the LLM found for it.

        Returns list of resolved entity IDs (passed to Call 2 as context).
        """
        from memforge.pipeline.entity_filter import filter_entities

        # Filter low-confidence or malformed entities
        raw_dicts = [
            {"name": e.name, "type": e.type, "tags": e.tags, "confidence": e.confidence} for e in enrichment.entities
        ]
        filtered_dicts, filter_stats = filter_entities(raw_dicts)
        filtered_names = {d["name"] for d in filtered_dicts}

        logger.info(
            "Entity filter: %d → %d (removed %d noise)",
            len(enrichment.entities),
            len(filtered_dicts),
            len(enrichment.entities) - len(filtered_dicts),
        )

        resolved_ids: list[int] = []

        for raw_entity in enrichment.entities:
            if raw_entity.name not in filtered_names:
                continue

            tags = (
                raw_entity.tags
                if raw_entity.tags
                else ([raw_entity.type] if raw_entity.type and raw_entity.type != "unknown" else [])
            )

            # Resolve: find existing entity or create new
            entity_id = await self.entity_resolver.resolve(
                extracted_name=raw_entity.name,
                db=self.db,
                extracted_tags=tags,
                doc_context=doc_context,
            )
            resolved_ids.append(entity_id)

            # Register aliases the LLM found for this entity
            for alias_name in raw_entity.aliases or []:
                await insert_llm_alias(
                    alias_name=alias_name,
                    canonical_name=raw_entity.name,
                    canonical_id=entity_id,
                    evidence="",
                    db=self.db,
                )

        self.entity_resolver.invalidate_cache()
        return resolved_ids

    # -------------------------------------------------------------------
    # Process Call 2: Build memories, dedup, insert
    # -------------------------------------------------------------------

    async def process_memories(
        self,
        doc_id: str,
        raw_memories: list[RawMemory],
        source_type: str,
        project_key: str | None = None,
        repo_identifier: str | None = None,
        entity_ids: list[int] | None = None,
        *,
        audit_context: Any | None = None,
        user_id: str | None = None,
        source_updated_at: datetime | None,
    ) -> dict:
        """Process extracted memories: build, dedup, and persist.

        Returns stats dict with counts of inserted, corroborated, skipped.
        """
        stats = {"inserted": 0, "corroborated": 0, "skipped": 0}

        for raw in raw_memories:
            if not self._candidate_can_persist(raw, stats):
                continue

            unit = await self._document_evidence_unit(
                doc_id=doc_id,
                raw=raw,
                source_type=source_type,
                project_key=project_key,
                repo_identifier=repo_identifier,
                extractor_run_id=getattr(audit_context, "run_id", None),
            )
            if await self._evidence_unit_has_materialized_memory(unit):
                stats["skipped"] += 1
                continue
            lifecycle = MemoryRelationApplyService().derive_lifecycle(unit, [])
            if lifecycle.action is not LifecycleAction.CREATE_MEMORY or not lifecycle.created_memory_id:
                await self._record_document_relation_outcome(
                    unit=unit,
                    relation_run_id=_document_relation_run_id(
                        unit,
                        LifecycleAction.CREATE_REVIEW,
                        relation_type=RelationType.SUPPORTS,
                    ),
                    lifecycle_action=lifecycle.action,
                    memory_id=None,
                    status="review",
                    review_case=lifecycle.review_case,
                    audit={
                        "source": "memory_engine.process_memories",
                        "review_case": lifecycle.review_case.value if lifecycle.review_case else None,
                    },
                )
                stats["skipped"] += 1
                continue

            # Build memory object
            memory = Memory(
                id=lifecycle.created_memory_id,
                memory_type=raw.memory_type,
                content=raw.content.strip(),
                content_hash=content_hash(raw.content.strip()),
                visibility=unit.visibility,
                owner_user_id=unit.owner_user_id,
                project_key=project_key,
                repo_identifier=repo_identifier,
                entity_refs=raw.entity_refs,
                tags=raw.tags,
                confidence=raw.confidence,
                corroboration_count=1,
                contradiction_count=0,
                valid_from=_parse_date(raw.valid_from),
                valid_until=_parse_date(raw.valid_until),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                status="active",
                extraction_context=raw.extraction_context,
            )

            # Resolve entity refs to IDs for linking
            # Filter first — Call 2's LLM sometimes outputs code names as entity_refs
            from memforge.pipeline.entity_filter import filter_entities

            ref_dicts = [{"name": name, "type": "unknown"} for name in raw.entity_refs]
            filtered_refs, _ = filter_entities(ref_dicts)
            filtered_ref_names = {d["name"] for d in filtered_refs}

            memory_entity_ids: list[int] = []
            for entity_name in raw.entity_refs:
                if entity_name not in filtered_ref_names:
                    continue
                try:
                    eid = await resolve_entity(entity_name, db=self.db)
                    memory_entity_ids.append(eid)
                except Exception as e:
                    logger.warning("Failed to resolve entity %r: %s", entity_name, e)

            # Dedup + insert (or corroborate)
            result = await self.memory_store.deduplicate_and_insert(
                memory=memory,
                doc_id=doc_id,
                source_type=source_type,
                entity_ids=memory_entity_ids,
                excerpt=raw.extraction_context,
                source_updated_at=source_updated_at,
                relation_outcome=self._document_relation_outcome_bundle(
                    unit=unit,
                    relation_run_id=_document_relation_run_id(
                        unit,
                        LifecycleAction.CREATE_MEMORY,
                        relation_type=RelationType.SUPPORTS,
                    ),
                    lifecycle_action=lifecycle.action,
                    memory_id=memory.id,
                    status="applied",
                    review_case=lifecycle.review_case,
                    audit={"source": "memory_engine.process_memories"},
                ),
            )

            if result == "inserted":
                stats["inserted"] += 1
                stats.setdefault("_inserted_ids", []).append(memory.id)
            elif result == "corroborated":
                stats["corroborated"] += 1
            else:
                stats["skipped"] += 1

        # Cross-document contradiction detection
        inserted_ids = stats.pop("_inserted_ids", [])
        if inserted_ids and self.structured_llm_client:
            from memforge.pipeline.contradiction_detector import detect_cross_doc_contradictions

            contradiction_stats = await detect_cross_doc_contradictions(
                new_memory_ids=inserted_ids,
                doc_id=doc_id,
                db=self.db,
                memory_store=self.memory_store,
                structured_llm_client=self.structured_llm_client,
                llm_model=self.llm_model,
                audit_context=audit_context,
                actor_user_id=user_id,
            )
            stats["contradictions_found"] = contradiction_stats.get("contradictions", 0)

        logger.info(
            "Memory processing for %s: %d inserted, %d corroborated, %d skipped",
            doc_id,
            stats["inserted"],
            stats["corroborated"],
            stats["skipped"],
        )
        return stats

    # -------------------------------------------------------------------
    # Process with reconciliation (for document UPDATES)
    # -------------------------------------------------------------------

    async def reconcile_and_persist(
        self,
        doc_id: str,
        raw_memories: list[RawMemory],
        source_type: str,
        doc_type: str,
        project_key: str | None = None,
        repo_identifier: str | None = None,
        entity_ids: list[int] | None = None,
        document_content: str | None = None,
        update_mode: str = "full_document",
        changed_hunks: str | None = None,
        update_plan_stats: dict[str, Any] | None = None,
        *,
        audit_context: Any | None = None,
        user_id: str | None = None,
        source_updated_at: datetime | None,
    ) -> dict:
        """Process memories with LLM reconciliation against existing memories.

        Called when a document is UPDATED (content hash changed). Compares
        new extractions against existing memories from the same source document,
        then executes ADD/UPDATE/SUPERSEDE/DELETE/NOOP operations.

        Returns stats dict with operation counts.
        """
        from memforge.pipeline.reconciler import reconcile_memories
        from memforge.pipeline.entity_filter import filter_entities

        stats = {
            "added": 0,
            "updated": 0,
            "superseded": 0,
            "deleted": 0,
            "noop": 0,
            "pending_review": 0,
            "skipped": 0,
        }

        # Same-document reconciliation can mutate only memories extracted by this document.
        existing = await self.db.get_memories_by_source_doc(doc_id, support_kind="extracted")
        existing_active = [m for m in existing if m.status == "active"]

        if not raw_memories and not existing_active:
            return stats

        # If no existing memories, skip reconciliation — just insert
        if not existing_active:
            result = await self.process_memories(
                doc_id=doc_id,
                raw_memories=raw_memories,
                source_type=source_type,
                project_key=project_key,
                repo_identifier=repo_identifier,
                entity_ids=entity_ids,
                audit_context=audit_context,
                user_id=user_id,
                source_updated_at=source_updated_at,
            )
            stats["added"] = result.get("inserted", 0)
            stats["skipped"] = result.get("skipped", 0)
            return stats

        # Filter entity_refs before reconciliation
        filtered_memories = []
        for raw in raw_memories:
            if not self._candidate_can_persist(raw, stats):
                continue
            ref_dicts = [{"name": name, "type": "unknown"} for name in raw.entity_refs]
            filtered_refs, _ = filter_entities(ref_dicts)
            raw.entity_refs = [d["name"] for d in filtered_refs]
            filtered_memories.append(raw)

        if not filtered_memories:
            if not document_content and any(raw.content.strip() for raw in raw_memories):
                await self._remove_filtered_document_support(existing_active, doc_id)
            if not self.structured_llm_client or not document_content:
                return stats

        # Call 3: LLM reconciliation
        if not self.structured_llm_client:
            logger.warning("No LLM client for reconciliation — falling back to deduplicate_and_insert")
            result = await self.process_memories(
                doc_id=doc_id,
                raw_memories=filtered_memories,
                source_type=source_type,
                project_key=project_key,
                repo_identifier=repo_identifier,
                entity_ids=entity_ids,
                audit_context=audit_context,
                user_id=user_id,
                source_updated_at=source_updated_at,
            )
            stats["added"] = result.get("inserted", 0)
            stats["skipped"] += result.get("skipped", 0)
            return stats

        llm_model = getattr(self, "llm_model", "claude-sonnet-4-20250514")
        reconciliation_result = await reconcile_memories(
            new_extractions=filtered_memories,
            existing_memories=existing_active,
            doc_type=doc_type,
            structured_llm_client=self.structured_llm_client,
            llm_model=llm_model,
            updated_document=document_content,
            update_mode=update_mode,
            changed_hunks=changed_hunks,
            update_plan_stats=update_plan_stats,
            include_metadata=True,
        )
        operations = getattr(reconciliation_result, "operations", reconciliation_result)
        failure = getattr(reconciliation_result, "failure", None)
        if failure:
            await self._record_reconciliation_failed(
                doc_id=doc_id,
                update_mode=update_mode,
                update_plan_stats=update_plan_stats,
                new_extraction_count=len(filtered_memories),
                existing_memory_count=len(existing_active),
                error_type=failure.error_type,
                error=failure.error,
            )
            stats["skipped"] += len(filtered_memories)
        existing_by_id = {m.id: m for m in existing_active}

        # Execute operations
        for op in operations:
            try:
                existing_memory = existing_by_id.get(op.memory_id or "")
                await self._record_reconciliation_decision(
                    op=op,
                    doc_id=doc_id,
                    update_mode=update_mode,
                    update_plan_stats=update_plan_stats,
                )

                if await self._operation_lacks_document_authority(op, existing_memory):
                    await self._record_reconciliation_authority_rejected(
                        op=op,
                        doc_id=doc_id,
                        update_mode=update_mode,
                        reason="Current document has no extracted support authority for this memory",
                    )
                    stats["skipped"] += 1
                    logger.info(
                        "RECONCILE AUTHORITY REJECTED: %s %s - current document has no extracted support",
                        op.action,
                        op.memory_id,
                    )
                    continue

                shared_support_reason = await self._shared_support_review_reason(op, doc_id)
                if shared_support_reason and existing_memory:
                    await self._record_reconciliation_review_gated(
                        op=op,
                        doc_id=doc_id,
                        update_mode=update_mode,
                        reason=shared_support_reason,
                    )
                    if op.memory:
                        if not await self._stage_replacement_review(
                            op=op,
                            existing_memory=existing_memory,
                            doc_id=doc_id,
                            source_type=source_type,
                            source_updated_at=source_updated_at,
                            project_key=project_key,
                            stats=stats,
                            user_id=user_id,
                            repo_identifier=repo_identifier,
                            audit_context=audit_context,
                        ):
                            continue
                    else:
                        await self.memory_store.mark_pending_review(
                            op.memory_id,
                            reason=shared_support_reason,
                            relation_outcome=await self._pending_review_relation_outcome(
                                op=op,
                                existing_memory=existing_memory,
                                doc_id=doc_id,
                                source_type=source_type,
                                project_key=project_key,
                                repo_identifier=repo_identifier,
                                user_id=user_id,
                                reason=shared_support_reason,
                                audit_context=audit_context,
                            ),
                        )
                    stats["pending_review"] += 1
                    logger.info(
                        "RECONCILE REVIEW: %s %s - %s",
                        op.action,
                        op.memory_id,
                        shared_support_reason,
                    )
                    continue

                if op.action == ReconcileAction.DELETE and op.memory_id:
                    if await self._delete_requires_review(op, doc_id, existing_memory):
                        await self._record_reconciliation_review_gated(
                            op=op,
                            doc_id=doc_id,
                            update_mode=update_mode,
                            reason=op.reason,
                        )
                        await self.memory_store.mark_pending_review(
                            op.memory_id,
                            reason=op.reason,
                            relation_outcome=await self._pending_review_relation_outcome(
                                op=op,
                                existing_memory=existing_memory,
                                doc_id=doc_id,
                                source_type=source_type,
                                project_key=project_key,
                                repo_identifier=repo_identifier,
                                user_id=user_id,
                                reason=op.reason,
                                audit_context=audit_context,
                            ),
                        )
                        stats["pending_review"] += 1
                        logger.info(
                            "RECONCILE REVIEW: %s %s - %s",
                            op.action,
                            op.memory_id,
                            op.reason,
                        )
                        continue

                    await self.memory_store.remove_source_support(op.memory_id, doc_id, reason="no_support")
                    stats["deleted"] += 1
                    logger.info("RECONCILE DELETE: %s - %s", op.memory_id, op.reason)
                    continue

                if requires_human_review(
                    op,
                    corroboration_count=existing_memory.corroboration_count if existing_memory else 0,
                ):
                    if (
                        op.action in (ReconcileAction.UPDATE, ReconcileAction.SUPERSEDE)
                        and op.memory
                        and existing_memory
                    ):
                        if not await self._stage_replacement_review(
                            op=op,
                            existing_memory=existing_memory,
                            doc_id=doc_id,
                            source_type=source_type,
                            source_updated_at=source_updated_at,
                            project_key=project_key,
                            stats=stats,
                            user_id=user_id,
                            repo_identifier=repo_identifier,
                            audit_context=audit_context,
                        ):
                            continue
                    elif op.memory_id and existing_memory:
                        await self.memory_store.mark_pending_review(
                            op.memory_id,
                            reason=op.reason,
                            relation_outcome=await self._pending_review_relation_outcome(
                                op=op,
                                existing_memory=existing_memory,
                                doc_id=doc_id,
                                source_type=source_type,
                                project_key=project_key,
                                repo_identifier=repo_identifier,
                                user_id=user_id,
                                reason=op.reason,
                                audit_context=audit_context,
                            ),
                        )
                    stats["pending_review"] += 1
                    logger.info(
                        "RECONCILE REVIEW: %s %s - %s",
                        op.action,
                        op.memory_id,
                        op.reason,
                    )
                    continue

                if op.action == ReconcileAction.ADD and op.memory:
                    if not self._candidate_can_persist(op.memory, stats):
                        continue
                    unit = await self._document_evidence_unit(
                        doc_id=doc_id,
                        raw=op.memory,
                        source_type=source_type,
                        project_key=project_key,
                        repo_identifier=repo_identifier,
                        extractor_run_id=getattr(audit_context, "run_id", None),
                    )
                    if await self._evidence_unit_has_materialized_memory(unit):
                        stats["noop"] += 1
                        continue
                    lifecycle = MemoryRelationApplyService().derive_lifecycle(unit, [])
                    if lifecycle.action is not LifecycleAction.CREATE_MEMORY or not lifecycle.created_memory_id:
                        await self._record_document_relation_outcome(
                            unit=unit,
                            relation_run_id=_document_relation_run_id(
                                unit,
                                LifecycleAction.CREATE_REVIEW,
                                relation_type=RelationType.SUPPORTS,
                            ),
                            lifecycle_action=lifecycle.action,
                            memory_id=None,
                            status="review",
                            review_case=lifecycle.review_case,
                            audit={
                                "source": "memory_engine.reconcile_and_persist",
                                "reconcile_action": op.action.value,
                                "review_case": lifecycle.review_case.value if lifecycle.review_case else None,
                            },
                        )
                        stats["pending_review"] += 1
                        continue
                    memory = self._build_memory(
                        op.memory,
                        project_key,
                        visibility=unit.visibility,
                        owner_user_id=unit.owner_user_id,
                        repo_identifier=repo_identifier,
                        memory_id=lifecycle.created_memory_id,
                    )
                    memory_entity_ids = await self._resolve_entity_refs(op.memory.entity_refs)
                    result = await self.memory_store.deduplicate_and_insert(
                        memory=memory,
                        doc_id=doc_id,
                        source_type=source_type,
                        entity_ids=memory_entity_ids,
                        excerpt=op.memory.extraction_context,
                        source_updated_at=source_updated_at,
                        relation_outcome=self._document_relation_outcome_bundle(
                            unit=unit,
                            relation_run_id=_document_relation_run_id(
                                unit,
                                LifecycleAction.CREATE_MEMORY,
                                relation_type=RelationType.SUPPORTS,
                            ),
                            lifecycle_action=lifecycle.action,
                            memory_id=memory.id,
                            status="applied",
                            review_case=lifecycle.review_case,
                            audit={
                                "source": "memory_engine.reconcile_and_persist",
                                "reconcile_action": op.action.value,
                            },
                        ),
                    )
                    if result == "inserted":
                        stats["added"] += 1
                        stats.setdefault("_inserted_ids", []).append(memory.id)

                elif op.action == ReconcileAction.UPDATE and op.memory_id and op.memory:
                    if not self._candidate_can_persist(op.memory, stats):
                        continue
                    (
                        unit,
                        lifecycle,
                        relation_run_id,
                        candidates,
                        incomplete_buckets,
                        candidate_count,
                    ) = await self._derive_document_replacement_lifecycle(
                        op=op,
                        doc_id=doc_id,
                        source_type=source_type,
                        project_key=project_key,
                        repo_identifier=repo_identifier,
                        user_id=user_id,
                        replacement_relation=RelationType.REFINES,
                        audit_context=audit_context,
                    )
                    if lifecycle.action is not LifecycleAction.SUPERSEDE_MEMORY:
                        await self._record_document_relation_outcome(
                            unit=unit,
                            relation_run_id=relation_run_id,
                            lifecycle_action=lifecycle.action,
                            memory_id=None,
                            status="review",
                            review_case=lifecycle.review_case,
                            audit={
                                "source": "memory_engine.reconcile_and_persist",
                                "reconcile_action": op.action.value,
                                "target_memory_id": op.memory_id,
                                "review_case": lifecycle.review_case.value if lifecycle.review_case else None,
                            },
                            candidates=candidates,
                            incomplete_mandatory_buckets=incomplete_buckets,
                            candidate_count=candidate_count,
                        )
                        stats["pending_review"] += 1
                        continue
                    new_memory = self._build_memory(
                        op.memory,
                        project_key,
                        visibility=unit.visibility,
                        owner_user_id=unit.owner_user_id,
                        repo_identifier=repo_identifier,
                    )
                    memory_entity_ids = await self._resolve_entity_refs(op.memory.entity_refs)
                    await self.memory_store.supersede_memory(
                        old_memory_id=op.memory_id,
                        new_memory=new_memory,
                        doc_id=doc_id,
                        source_type=source_type,
                        entity_ids=memory_entity_ids,
                        excerpt=op.memory.extraction_context,
                        replacement_reason=op.reason,
                        replacement_kind="revision",
                        source_updated_at=source_updated_at,
                        relation_outcome=self._document_relation_outcome_bundle(
                            unit=unit,
                            relation_run_id=relation_run_id,
                            lifecycle_action=lifecycle.action,
                            memory_id=new_memory.id,
                            status="applied",
                            review_case=lifecycle.review_case,
                            audit={
                                "source": "memory_engine.reconcile_and_persist",
                                "reconcile_action": op.action.value,
                                "target_memory_id": op.memory_id,
                            },
                            candidates=candidates,
                            incomplete_mandatory_buckets=incomplete_buckets,
                            candidate_count=candidate_count,
                        ),
                    )
                    stats["updated"] += 1
                    logger.info("RECONCILE UPDATE: %s -> %s - %s", op.memory_id, new_memory.id, op.reason)

                elif op.action == ReconcileAction.SUPERSEDE and op.memory_id and op.memory:
                    if not self._candidate_can_persist(op.memory, stats):
                        continue
                    (
                        unit,
                        lifecycle,
                        relation_run_id,
                        candidates,
                        incomplete_buckets,
                        candidate_count,
                    ) = await self._derive_document_replacement_lifecycle(
                        op=op,
                        doc_id=doc_id,
                        source_type=source_type,
                        project_key=project_key,
                        repo_identifier=repo_identifier,
                        user_id=user_id,
                        replacement_relation=RelationType.CONTRADICTS,
                        audit_context=audit_context,
                    )
                    if lifecycle.action is not LifecycleAction.SUPERSEDE_MEMORY:
                        await self._record_document_relation_outcome(
                            unit=unit,
                            relation_run_id=relation_run_id,
                            lifecycle_action=lifecycle.action,
                            memory_id=None,
                            status="review",
                            review_case=lifecycle.review_case,
                            audit={
                                "source": "memory_engine.reconcile_and_persist",
                                "reconcile_action": op.action.value,
                                "target_memory_id": op.memory_id,
                                "review_case": lifecycle.review_case.value if lifecycle.review_case else None,
                            },
                            candidates=candidates,
                            incomplete_mandatory_buckets=incomplete_buckets,
                            candidate_count=candidate_count,
                        )
                        stats["pending_review"] += 1
                        continue
                    new_memory = self._build_memory(
                        op.memory,
                        project_key,
                        visibility=unit.visibility,
                        owner_user_id=unit.owner_user_id,
                        repo_identifier=repo_identifier,
                    )
                    memory_entity_ids = await self._resolve_entity_refs(op.memory.entity_refs)
                    await self.memory_store.supersede_memory(
                        old_memory_id=op.memory_id,
                        new_memory=new_memory,
                        doc_id=doc_id,
                        source_type=source_type,
                        entity_ids=memory_entity_ids,
                        excerpt=op.memory.extraction_context,
                        replacement_reason=op.reason,
                        replacement_kind="supersession",
                        source_updated_at=source_updated_at,
                        relation_outcome=self._document_relation_outcome_bundle(
                            unit=unit,
                            relation_run_id=relation_run_id,
                            lifecycle_action=lifecycle.action,
                            memory_id=new_memory.id,
                            status="applied",
                            review_case=lifecycle.review_case,
                            audit={
                                "source": "memory_engine.reconcile_and_persist",
                                "reconcile_action": op.action.value,
                                "target_memory_id": op.memory_id,
                            },
                            candidates=candidates,
                            incomplete_mandatory_buckets=incomplete_buckets,
                            candidate_count=candidate_count,
                        ),
                    )
                    stats["superseded"] += 1
                    logger.info("RECONCILE SUPERSEDE: %s -> %s - %s", op.memory_id, new_memory.id, op.reason)

                else:
                    stats["noop"] += 1

            except Exception as e:
                logger.error("Reconciliation action %s failed: %s", op.action, e)
                await self.memory_store.record_audit_event(
                    "reconciliation_action_failed",
                    "failed",
                    memory_id=op.memory_id,
                    doc_id=doc_id,
                    decision=op.action.value if hasattr(op.action, "value") else str(op.action),
                    reason=op.reason,
                    error=str(e),
                    payload={"action": op.action.value if hasattr(op.action, "value") else str(op.action)},
                )
                stats["skipped"] += 1

        # Cross-document contradiction detection for newly added memories
        inserted_ids = stats.pop("_inserted_ids", [])
        if inserted_ids and self.structured_llm_client:
            from memforge.pipeline.contradiction_detector import detect_cross_doc_contradictions

            contradiction_stats = await detect_cross_doc_contradictions(
                new_memory_ids=inserted_ids,
                doc_id=doc_id,
                db=self.db,
                memory_store=self.memory_store,
                structured_llm_client=self.structured_llm_client,
                llm_model=self.llm_model,
                audit_context=audit_context,
                actor_user_id=user_id,
            )
            stats["contradictions_found"] = contradiction_stats.get("contradictions", 0)

        logger.info(
            "Reconciliation for %s: +%d added, ~%d updated, %d superseded, -%d deleted, =%d noop",
            doc_id,
            stats["added"],
            stats["updated"],
            stats["superseded"],
            stats["deleted"],
            stats["noop"],
        )
        return stats

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    async def _remove_filtered_document_support(
        self,
        existing_active: list[Memory],
        doc_id: str,
    ) -> None:
        """Remove this document as support when its update produced no usable candidates."""
        for memory in existing_active:
            await self.memory_store.remove_source_support(memory.id, doc_id, reason="no_support")

    async def _delete_requires_review(
        self,
        op: ReconcileOperation,
        doc_id: str,
        existing_memory: Memory | None,
    ) -> bool:
        """Review DELETE only when it is flagged or would retire a well-supported memory."""
        if op.flag_for_review:
            return True
        if not existing_memory or existing_memory.corroboration_count < 3 or not op.memory_id:
            return False

        sources = await self.db.get_memory_sources(op.memory_id)
        remaining_sources = [source for source in sources if source.doc_id != doc_id]
        return not remaining_sources

    async def _operation_lacks_document_authority(
        self,
        op: ReconcileOperation,
        existing_memory: Memory | None,
    ) -> bool:
        """Reject model-proposed lifecycle writes outside current-doc extracted ownership."""
        if op.action not in (ReconcileAction.UPDATE, ReconcileAction.SUPERSEDE, ReconcileAction.DELETE):
            return False
        return bool(op.memory_id and existing_memory is None)

    async def _shared_support_review_reason(
        self,
        op: ReconcileOperation,
        doc_id: str,
    ) -> str | None:
        """Route content mutations to review when other support edges exist.

        Supporter verification can later prove compatibility and allow direct mutation.
        Until then, the lifecycle path stays conservative.
        """
        if op.action not in (ReconcileAction.UPDATE, ReconcileAction.SUPERSEDE) or not op.memory_id:
            return None
        sources = await self.db.get_memory_sources(op.memory_id)
        other_sources = [source for source in sources if source.doc_id != doc_id]
        if not other_sources:
            return None
        kinds = sorted({source.support_kind for source in other_sources})
        return (
            "Content mutation requires review because other support edges still "
            f"exist for this memory: {', '.join(kinds)}"
        )

    async def _stage_replacement_review(
        self,
        *,
        op: ReconcileOperation,
        existing_memory: Memory,
        doc_id: str,
        source_type: str,
        source_updated_at: datetime | None,
        project_key: str | None,
        stats: dict,
        user_id: str | None = None,
        repo_identifier: str | None = None,
        audit_context: Any | None = None,
    ) -> bool:
        """Insert a hidden challenger and create a review case for a replacement."""
        if not op.memory or not self._candidate_can_persist(op.memory, stats):
            return False
        is_revision = op.action == ReconcileAction.UPDATE
        replacement_relation = RelationType.REFINES if is_revision else RelationType.CONTRADICTS
        replacement_kind: ReplacementKind = "revision" if is_revision else "supersession"
        (
            unit,
            _lifecycle,
            relation_run_id,
            candidates,
            incomplete_buckets,
            candidate_count,
        ) = await self._derive_document_replacement_lifecycle(
            op=op,
            doc_id=doc_id,
            source_type=source_type,
            project_key=project_key,
            repo_identifier=repo_identifier,
            user_id=user_id,
            replacement_relation=replacement_relation,
            audit_context=audit_context,
            lifecycle_action=LifecycleAction.CREATE_REVIEW,
        )
        challenger = self._build_memory(
            op.memory,
            project_key,
            visibility=unit.visibility,
            owner_user_id=unit.owner_user_id,
        )
        challenger.status = "pending_review"
        memory_entity_ids = await self._resolve_entity_refs(op.memory.entity_refs)
        existing_case = await self.db.get_open_review_for_incumbent_source_doc(
            incumbent_memory_id=existing_memory.id,
            doc_id=doc_id,
            kind=ReviewKind.SUPERSEDE.value,
        )
        review = None
        related_review_id = existing_case.id if existing_case else None
        if existing_case is None:
            review = MemoryReview(
                id=generate_deterministic_review_id(
                    kind=ReviewKind.SUPERSEDE.value,
                    incumbent_memory_id=existing_memory.id,
                    challenger_memory_id=challenger.id,
                ),
                kind=ReviewKind.SUPERSEDE.value,
                status=ReviewStatus.PENDING.value,
                incumbent_memory_id=existing_memory.id,
                challenger_memory_id=challenger.id,
                reason=op.reason,
                expected_incumbent_updated_at=(
                    existing_memory.updated_at.isoformat() if existing_memory.updated_at else None
                ),
                expected_challenger_updated_at=(challenger.updated_at.isoformat() if challenger.updated_at else None),
                replacement_kind=replacement_kind,
                created_at=datetime.now(timezone.utc),
            )
        await self.memory_store.insert_memory(
            memory=challenger,
            doc_id=doc_id,
            source_type=source_type,
            entity_ids=memory_entity_ids,
            excerpt=op.memory.extraction_context,
            source_updated_at=source_updated_at,
            relation_outcome=self._document_relation_outcome_bundle(
                unit=unit,
                relation_run_id=relation_run_id,
                lifecycle_action=LifecycleAction.CREATE_REVIEW,
                memory_id=challenger.id,
                status="review",
                review_case=ReviewCase.MANUAL_REVIEW_GATE,
                audit={
                    "source": "memory_engine.reconcile_and_persist",
                    "reconcile_action": op.action.value,
                    "target_memory_id": op.memory_id,
                    "manual_gate_reason": op.reason,
                },
                candidates=candidates,
                incomplete_mandatory_buckets=incomplete_buckets,
                candidate_count=candidate_count,
            ),
            review=review,
            related_review_id=related_review_id,
            related_review_reason=op.reason,
        )
        return True

    async def _record_reconciliation_decision(
        self,
        *,
        op: ReconcileOperation,
        doc_id: str,
        update_mode: str,
        update_plan_stats: dict[str, Any] | None,
    ) -> None:
        """Audit the model's reconciliation decision before applying it."""
        if not hasattr(self.memory_store, "record_audit_event"):
            return
        context = None
        if hasattr(self.memory_store, "operation_context"):
            context = self.memory_store.operation_context(doc_id=doc_id)
        payload = {
            "update_mode": update_mode,
            "has_candidate": op.memory is not None,
        }
        if update_plan_stats:
            payload["update_plan_stats"] = update_plan_stats
        await self.memory_store.record_audit_event(
            "reconciliation_decision_returned",
            "committed",
            context=context,
            doc_id=doc_id,
            memory_id=op.memory_id,
            decision=op.action.value if hasattr(op.action, "value") else str(op.action),
            reason=op.reason,
            support_kind="extracted",
            payload=payload,
        )

    async def _record_reconciliation_failed(
        self,
        *,
        doc_id: str,
        update_mode: str,
        update_plan_stats: dict[str, Any] | None,
        new_extraction_count: int,
        existing_memory_count: int,
        error_type: str,
        error: str,
    ) -> None:
        """Audit a reconciliation call that failed before lifecycle decisions were returned."""
        if not hasattr(self.memory_store, "record_audit_event"):
            return
        context = None
        if hasattr(self.memory_store, "operation_context"):
            context = self.memory_store.operation_context(doc_id=doc_id)
        payload = {
            "update_mode": update_mode,
            "new_extraction_count": new_extraction_count,
            "existing_memory_count": existing_memory_count,
        }
        if update_plan_stats:
            payload["update_plan_stats"] = update_plan_stats
        await self.memory_store.record_audit_event(
            "reconciliation_failed",
            "failed",
            context=context,
            doc_id=doc_id,
            decision="skip_mutations",
            reason=error_type,
            support_kind="extracted",
            payload=payload,
            error=error,
        )

    async def _record_reconciliation_authority_rejected(
        self,
        *,
        op: ReconcileOperation,
        doc_id: str,
        update_mode: str,
        reason: str,
    ) -> None:
        """Audit a model operation that tried to mutate outside document authority."""
        if not hasattr(self.memory_store, "record_audit_event"):
            return
        context = None
        if hasattr(self.memory_store, "operation_context"):
            context = self.memory_store.operation_context(doc_id=doc_id)
        await self.memory_store.record_audit_event(
            "reconciliation_authority_rejected",
            "committed",
            context=context,
            doc_id=doc_id,
            memory_id=op.memory_id,
            decision=op.action.value if hasattr(op.action, "value") else str(op.action),
            reason=reason,
            support_kind="extracted",
            payload={"update_mode": update_mode, "has_candidate": op.memory is not None},
        )

    async def _record_reconciliation_review_gated(
        self,
        *,
        op: ReconcileOperation,
        doc_id: str,
        update_mode: str,
        reason: str,
    ) -> None:
        """Audit a reconciliation operation that was routed to human review."""
        if not hasattr(self.memory_store, "record_audit_event"):
            return
        context = None
        if hasattr(self.memory_store, "operation_context"):
            context = self.memory_store.operation_context(doc_id=doc_id)
        await self.memory_store.record_audit_event(
            "reconciliation_review_gated",
            "committed",
            context=context,
            doc_id=doc_id,
            memory_id=op.memory_id,
            decision=op.action.value if hasattr(op.action, "value") else str(op.action),
            reason=reason,
            support_kind="extracted",
            payload={"update_mode": update_mode},
        )

    def _candidate_can_persist(self, raw: RawMemory, stats: dict | None = None) -> bool:
        """Return whether a raw candidate should be persisted, updating stats when skipped."""
        quality = classify_memory_candidate(raw)
        if quality.keep:
            return True

        if stats is not None:
            stats["skipped"] = stats.get("skipped", 0) + 1
        logger.info(
            "Skipping memory candidate (%s): %s",
            quality.skip_reason,
            raw.content.strip()[:120],
        )
        return False

    def _build_memory(
        self,
        raw: RawMemory,
        project_key: str | None,
        *,
        visibility: str,
        owner_user_id: str | None,
        repo_identifier: str | None = None,
        memory_id: str | None = None,
    ) -> Memory:
        """Build a Memory object from a RawMemory."""
        return Memory(
            id=memory_id or generate_memory_id(),
            memory_type=raw.memory_type,
            content=raw.content.strip(),
            content_hash=content_hash(raw.content.strip()),
            visibility=visibility,
            owner_user_id=owner_user_id,
            project_key=project_key,
            repo_identifier=repo_identifier,
            entity_refs=raw.entity_refs,
            tags=raw.tags,
            confidence=raw.confidence,
            corroboration_count=1,
            contradiction_count=0,
            valid_from=_parse_date(raw.valid_from),
            valid_until=_parse_date(raw.valid_until),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            status="active",
            extraction_context=raw.extraction_context,
        )

    async def _resolve_entity_refs(self, entity_refs: list[str]) -> list[int]:
        """Resolve entity names to IDs."""
        ids: list[int] = []
        for name in entity_refs:
            try:
                eid = await resolve_entity(name, db=self.db)
                ids.append(eid)
            except Exception as e:
                logger.warning("Failed to resolve entity %r: %s", name, e)
        return ids

    async def _document_evidence_unit(
        self,
        *,
        doc_id: str,
        raw: RawMemory,
        source_type: str,
        project_key: str | None,
        repo_identifier: str | None,
        extractor_run_id: str | None,
    ) -> EvidenceUnit:
        document = await self.db.get_document(doc_id)
        doc_revision_id = None
        source_id = source_type
        if document is not None:
            doc_revision_id = document.content_hash or document.version
            source_id = document.source or source_type
        visibility, owner_user_id = await memory_visibility_for_document(
            self.db,
            doc_id=doc_id,
        )
        content = raw.content.strip()
        evidence_id = _document_evidence_unit_id(
            source_id=source_id,
            doc_id=doc_id,
            doc_revision_id=doc_revision_id,
            source_type=source_type,
            content=content,
        )
        return EvidenceUnit(
            id=evidence_id,
            source_id=source_id,
            doc_id=doc_id,
            doc_revision_id=doc_revision_id,
            source_type=source_type,
            source_anchor=evidence_id,
            source_lineage_id=doc_id,
            project_key=project_key,
            visibility=visibility,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            content=content,
            excerpt=raw.extraction_context,
            evidence_provenance=(
                EvidenceContentProvenance.SOURCE_EXCERPT
                if raw.extraction_context
                else EvidenceContentProvenance.NO_EXCERPT
            ),
            source_metadata={
                "memory_type": raw.memory_type,
                "content_hash": content_hash(content),
            },
            observed_at=datetime.now(timezone.utc).isoformat(),
            extractor_run_id=extractor_run_id,
        )

    async def _evidence_unit_has_materialized_memory(self, unit: EvidenceUnit) -> bool:
        return await self.db.has_materialized_evidence_unit(unit.id)

    async def _derive_document_replacement_lifecycle(
        self,
        *,
        op: ReconcileOperation,
        doc_id: str,
        source_type: str,
        project_key: str | None,
        repo_identifier: str | None,
        user_id: str | None,
        replacement_relation: RelationType,
        audit_context: Any | None,
        lifecycle_action: LifecycleAction = LifecycleAction.SUPERSEDE_MEMORY,
    ) -> tuple[EvidenceUnit, LifecycleDecision, str, list[RelationCandidateRecord], tuple[str, ...], int]:
        assert op.memory is not None
        assert op.memory_id is not None
        unit = await self._document_evidence_unit(
            doc_id=doc_id,
            raw=op.memory,
            source_type=source_type,
            project_key=project_key,
            repo_identifier=repo_identifier,
            extractor_run_id=getattr(audit_context, "run_id", None),
        )
        relation_run_id = _document_relation_run_id(
            unit,
            lifecycle_action,
            candidate_memory_id=op.memory_id,
            relation_type=replacement_relation,
            authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
        )
        buckets = await build_mandatory_candidate_bucket_results(
            store=self.db,
            unit=unit,
            access_context=AccessContext(
                actor_user_id=user_id,
                source_subscriptions=(unit.source_id,),
                repo_identifier=repo_identifier,
                operation_type="document_replacement",
            ),
        )
        universe = build_candidate_universe(
            relation_run_id=relation_run_id,
            evidence_unit_id=unit.id,
            bucket_results=buckets,
        )
        target_candidate = next(
            (candidate for candidate in universe.candidates if candidate.memory_id == op.memory_id),
            None,
        )
        if target_candidate is None:
            raise RuntimeError("document replacement target missing from mandatory candidate universe")
        decision = RelationDecision(
            candidate_memory_id=op.memory_id,
            relation_type=replacement_relation,
            authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
            confidence=op.memory.confidence,
            reason=op.reason,
            proposed_memory_content=op.memory.content,
            evidence_excerpt=op.memory.extraction_context,
            matched_bucket=target_candidate.bucket,
            matched_bucket_complete=target_candidate.bucket_complete,
            classifier_batch_key=relation_run_id,
        )
        lifecycle = MemoryRelationApplyService().derive_lifecycle(unit, [decision])
        return (
            unit,
            lifecycle,
            relation_run_id,
            list(universe.candidates),
            universe.incomplete_mandatory_buckets,
            universe.total_unique_candidates,
        )

    async def _record_document_relation_outcome(
        self,
        *,
        unit: EvidenceUnit,
        relation_run_id: str,
        lifecycle_action: LifecycleAction,
        memory_id: str | None,
        status: str,
        review_case,
        audit: dict[str, object],
        candidates: list[RelationCandidateRecord] | None = None,
        incomplete_mandatory_buckets: tuple[str, ...] = (),
        candidate_count: int | None = None,
    ) -> None:
        await self.db.record_relation_outcome_bundle(
            self._document_relation_outcome_bundle(
                unit=unit,
                relation_run_id=relation_run_id,
                lifecycle_action=lifecycle_action,
                memory_id=memory_id,
                status=status,
                review_case=review_case,
                audit=audit,
                candidates=candidates,
                incomplete_mandatory_buckets=incomplete_mandatory_buckets,
                candidate_count=candidate_count,
            )
        )

    def _document_relation_outcome_bundle(
        self,
        *,
        unit: EvidenceUnit,
        relation_run_id: str,
        lifecycle_action: LifecycleAction,
        memory_id: str | None,
        status: str,
        review_case,
        audit: dict[str, object],
        candidates: list[RelationCandidateRecord] | None = None,
        incomplete_mandatory_buckets: tuple[str, ...] = (),
        candidate_count: int | None = None,
    ) -> RelationOutcomeBundle:
        candidates = candidates or []
        run_audit = dict(audit)
        relations: tuple[EvidenceRelationRecord, ...] = ()
        if memory_id is not None:
            relations = (
                EvidenceRelationRecord(
                    evidence_unit_id=unit.id,
                    memory_id=memory_id,
                    relation_type=RelationType.SUPPORTS,
                    authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
                    is_authoritative_support=True,
                    source_lineage_id=unit.source_lineage_id,
                    confidence=1.0,
                    reason="Memory created from this source evidence unit.",
                    excerpt=unit.excerpt,
                    classifier_version="memory-engine-v1",
                    relation_run_id=relation_run_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                ),
            )
        return RelationOutcomeBundle(
            evidence_unit=unit,
            relation_run=RelationRunRecord(
                id=relation_run_id,
                evidence_unit_id=unit.id,
                access_context_hash=unit.access_context_hash,
                candidate_count=len(candidates) if candidate_count is None else candidate_count,
                mandatory_candidate_count=sum(1 for candidate in candidates if candidate.is_mandatory),
                checked_candidate_count=sum(1 for candidate in candidates if candidate.was_checked),
                incomplete_mandatory_buckets=incomplete_mandatory_buckets,
                classifier_version="memory-engine-v1",
                lifecycle_action=lifecycle_action,
                review_case=review_case,
                status=status,
                result_memory_id=memory_id,
                audit=run_audit,
            ),
            candidates=tuple(candidates),
            relations=relations,
        )

    async def _pending_review_relation_outcome(
        self,
        *,
        op: ReconcileOperation,
        existing_memory: Memory,
        doc_id: str,
        source_type: str,
        project_key: str | None,
        repo_identifier: str | None,
        user_id: str | None,
        reason: str | None,
        audit_context: Any | None,
    ) -> RelationOutcomeBundle:
        if op.memory_id is None:
            raise RuntimeError("pending-review relation outcome requires a target memory id")

        review_observation = (reason or f"Current document requires review for memory {op.memory_id}").strip()
        raw = op.memory or RawMemory(
            content=review_observation,
            memory_type=existing_memory.memory_type,
            confidence=existing_memory.confidence,
            extraction_context=None,
        )
        unit = await self._document_evidence_unit(
            doc_id=doc_id,
            raw=raw,
            source_type=source_type,
            project_key=project_key,
            repo_identifier=repo_identifier,
            extractor_run_id=getattr(audit_context, "run_id", None),
        )
        relation_type = RelationType.REFINES if op.action == ReconcileAction.UPDATE else RelationType.CONTRADICTS
        relation_run_id = _document_relation_run_id(
            unit,
            LifecycleAction.CREATE_REVIEW,
            candidate_memory_id=op.memory_id,
            relation_type=relation_type,
            authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
        )
        buckets = await build_mandatory_candidate_bucket_results(
            store=self.db,
            unit=unit,
            access_context=AccessContext(
                actor_user_id=user_id,
                source_subscriptions=(unit.source_id,),
                repo_identifier=repo_identifier,
                operation_type="document_review",
            ),
        )
        universe = build_candidate_universe(
            relation_run_id=relation_run_id,
            evidence_unit_id=unit.id,
            bucket_results=buckets,
        )
        target_candidate = next(
            (candidate for candidate in universe.candidates if candidate.memory_id == op.memory_id),
            None,
        )
        if target_candidate is None:
            raise RuntimeError("pending-review target missing from mandatory candidate universe")

        relation = EvidenceRelationRecord(
            evidence_unit_id=unit.id,
            memory_id=op.memory_id,
            relation_type=relation_type,
            authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
            is_authoritative_support=True,
            source_lineage_id=unit.source_lineage_id,
            confidence=raw.confidence,
            reason=reason,
            proposed_memory_content=raw.content if op.memory else None,
            excerpt=unit.excerpt,
            classifier_version="memory-engine-v1",
            relation_run_id=relation_run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return RelationOutcomeBundle(
            evidence_unit=unit,
            relation_run=RelationRunRecord(
                id=relation_run_id,
                evidence_unit_id=unit.id,
                access_context_hash=unit.access_context_hash,
                candidate_count=universe.total_unique_candidates,
                mandatory_candidate_count=universe.mandatory_candidate_count,
                checked_candidate_count=universe.checked_candidate_count,
                incomplete_mandatory_buckets=universe.incomplete_mandatory_buckets,
                classifier_version="memory-engine-v1",
                lifecycle_action=LifecycleAction.CREATE_REVIEW,
                review_case=ReviewCase.MANUAL_REVIEW_GATE,
                status="review",
                result_memory_id=op.memory_id,
                audit={
                    "action": op.action.value,
                    "reason": reason,
                    "target_memory_id": op.memory_id,
                    "relation_type": relation_type.value,
                    "candidate_memory_ids": [candidate.memory_id for candidate in universe.candidates],
                },
            ),
            candidates=tuple(universe.candidates),
            relations=(relation,),
        )


def _document_evidence_unit_id(
    *,
    source_id: str,
    doc_id: str,
    doc_revision_id: str | None,
    source_type: str,
    content: str,
) -> str:
    digest = sha256(
        "\x1f".join([source_id, doc_id, doc_revision_id or "", source_type, content_hash(content)]).encode("utf-8")
    ).hexdigest()[:16]
    return f"eu-doc-{digest}"


def _document_relation_run_id(
    unit: EvidenceUnit,
    action: str | LifecycleAction,
    *,
    classifier_version: str = "memory-engine-v1",
    candidate_memory_id: str | None = None,
    relation_type: RelationType | None = None,
    authority_case: AuthorityCase | None = None,
) -> str:
    return relation_run_id_for(
        prefix="doc",
        unit=unit,
        action=action,
        classifier_version=classifier_version,
        candidate_memory_id=candidate_memory_id,
        relation_type=relation_type,
        authority_case=authority_case,
    )


def _parse_date(value: str | None) -> date | None:
    """Parse an ISO date or datetime string into a calendar date."""
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except (ValueError, TypeError):
        return None
