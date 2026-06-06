"""Memory Engine — orchestrates entity resolution, dedup, and storage.

This is the bridge between the enrichment/extraction pipeline and the memory store.
It processes enrichment results (Call 1) and extracted memories (Call 2) into
persisted, deduplicated, entity-linked memories.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memforge.memory.entity_resolver import EntityResolver, insert_llm_alias, resolve_entity
from memforge.memory.lifecycle import requires_human_review
from memforge.memory.quality import classify_memory_candidate
from memforge.memory.visibility_policy import default_visibility
from memforge.models import (
    Memory,
    MemoryReview,
    RawMemory,
    ReconcileAction,
    ReconcileOperation,
    ReviewKind,
    ReviewStatus,
    content_hash,
    generate_memory_id,
    generate_review_id,
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
        entity_ids: list[int] | None = None,
        audit_context: Any | None = None,
        user_id: str | None = None,
    ) -> dict:
        """Process extracted memories: build, dedup, and persist.

        Returns stats dict with counts of inserted, corroborated, skipped.
        """
        stats = {"inserted": 0, "corroborated": 0, "skipped": 0}

        for raw in raw_memories:
            if not self._candidate_can_persist(raw, stats):
                continue

            # Build memory object
            visibility, owner_user_id = default_visibility(source_type, user_id=user_id)
            memory = Memory(
                id=generate_memory_id(),
                memory_type=raw.memory_type,
                content=raw.content.strip(),
                content_hash=content_hash(raw.content.strip()),
                visibility=visibility,
                owner_user_id=owner_user_id,
                project_key=project_key,
                entity_refs=raw.entity_refs,
                tags=raw.tags,
                confidence=raw.confidence,
                corroboration_count=1,
                contradiction_count=0,
                valid_from=_parse_datetime(raw.valid_from),
                valid_until=_parse_datetime(raw.valid_until),
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
        entity_ids: list[int] | None = None,
        document_content: str | None = None,
        update_mode: str = "full_document",
        changed_hunks: str | None = None,
        update_plan_stats: dict[str, Any] | None = None,
        audit_context: Any | None = None,
        user_id: str | None = None,
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
                entity_ids=entity_ids,
                audit_context=audit_context,
                user_id=user_id,
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
                entity_ids=entity_ids,
                audit_context=audit_context,
                user_id=user_id,
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
                            project_key=project_key,
                            stats=stats,
                            user_id=user_id,
                        ):
                            continue
                    else:
                        await self.memory_store.mark_pending_review(op.memory_id, reason=shared_support_reason)
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
                        await self.memory_store.mark_pending_review(op.memory_id, reason=op.reason)
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
                    if op.action in (ReconcileAction.UPDATE, ReconcileAction.SUPERSEDE) and op.memory and existing_memory:
                        if not await self._stage_replacement_review(
                            op=op,
                            existing_memory=existing_memory,
                            doc_id=doc_id,
                            source_type=source_type,
                            project_key=project_key,
                            stats=stats,
                            user_id=user_id,
                        ):
                            continue
                    elif op.memory_id:
                        await self.memory_store.mark_pending_review(op.memory_id, reason=op.reason)
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
                    memory = self._build_memory(op.memory, project_key, source_type, user_id=user_id)
                    memory_entity_ids = await self._resolve_entity_refs(op.memory.entity_refs)
                    result = await self.memory_store.deduplicate_and_insert(
                        memory=memory,
                        doc_id=doc_id,
                        source_type=source_type,
                        entity_ids=memory_entity_ids,
                        excerpt=op.memory.extraction_context,
                    )
                    if result == "inserted":
                        stats["added"] += 1
                        stats.setdefault("_inserted_ids", []).append(memory.id)

                elif op.action == ReconcileAction.UPDATE and op.memory_id and op.memory:
                    if not self._candidate_can_persist(op.memory, stats):
                        continue
                    await self.memory_store.update_memory(
                        memory_id=op.memory_id,
                        new_content=op.memory.content,
                        new_confidence=op.memory.confidence,
                    )
                    stats["updated"] += 1
                    logger.info("RECONCILE UPDATE: %s - %s", op.memory_id, op.reason)

                elif op.action == ReconcileAction.SUPERSEDE and op.memory_id and op.memory:
                    if not self._candidate_can_persist(op.memory, stats):
                        continue
                    new_memory = self._build_memory(op.memory, project_key, source_type, user_id=user_id)
                    memory_entity_ids = await self._resolve_entity_refs(op.memory.entity_refs)
                    await self.memory_store.supersede_memory(
                        old_memory_id=op.memory_id,
                        new_memory=new_memory,
                        doc_id=doc_id,
                        source_type=source_type,
                        entity_ids=memory_entity_ids,
                        replacement_reason=op.reason,
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
        project_key: str | None,
        stats: dict,
        user_id: str | None = None,
    ) -> bool:
        """Insert a hidden challenger and create a review case for a replacement."""
        if not op.memory or not self._candidate_can_persist(op.memory, stats):
            return False
        challenger = self._build_memory(op.memory, project_key, source_type, user_id=user_id)
        challenger.status = "pending_review"
        memory_entity_ids = await self._resolve_entity_refs(op.memory.entity_refs)
        await self.memory_store.insert_memory(
            memory=challenger,
            doc_id=doc_id,
            source_type=source_type,
            entity_ids=memory_entity_ids,
            excerpt=op.memory.extraction_context,
        )
        await self._record_supersede_review(
            incumbent=existing_memory,
            challenger_id=challenger.id,
            reason=op.reason,
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
        source_type: str,
        user_id: str | None = None,
    ) -> Memory:
        """Build a Memory object from a RawMemory."""
        visibility, owner_user_id = default_visibility(source_type, user_id=user_id)
        return Memory(
            id=generate_memory_id(),
            memory_type=raw.memory_type,
            content=raw.content.strip(),
            content_hash=content_hash(raw.content.strip()),
            visibility=visibility,
            owner_user_id=owner_user_id,
            project_key=project_key,
            entity_refs=raw.entity_refs,
            tags=raw.tags,
            confidence=raw.confidence,
            corroboration_count=1,
            contradiction_count=0,
            valid_from=_parse_datetime(raw.valid_from),
            valid_until=_parse_datetime(raw.valid_until),
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

    async def _record_supersede_review(
        self,
        *,
        incumbent: Memory,
        challenger_id: str,
        reason: str | None,
    ) -> None:
        """Create a pending review tying an incumbent to a quarantined challenger.

        The review pins the incumbent's ``updated_at`` and the challenger's
        freshly-inserted state so later approval can detect drift.
        """
        existing = await self.db.get_pending_review_for_challenger(challenger_id)
        if existing:
            return

        for source in await self.db.get_memory_sources(challenger_id):
            existing_case = await self.db.get_open_review_for_incumbent_source_doc(
                incumbent_memory_id=incumbent.id,
                doc_id=source.doc_id,
                kind=ReviewKind.SUPERSEDE.value,
            )
            if existing_case:
                await self.db.add_memory_review_related_challenger(
                    existing_case.id,
                    challenger_id,
                    reason=reason,
                )
                return

        challenger = await self.db.get_memory(challenger_id)
        review = MemoryReview(
            id=generate_review_id(),
            kind=ReviewKind.SUPERSEDE.value,
            status=ReviewStatus.PENDING.value,
            incumbent_memory_id=incumbent.id,
            challenger_memory_id=challenger_id,
            reason=reason,
            expected_incumbent_updated_at=(incumbent.updated_at.isoformat() if incumbent.updated_at else None),
            expected_challenger_updated_at=(
                challenger.updated_at.isoformat() if challenger and challenger.updated_at else None
            ),
            created_at=datetime.now(timezone.utc),
        )
        await self.db.insert_memory_review(review)


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO datetime string, returning None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
