"""Memory Store - SQLite + ChromaDB + FTS5 synchronized storage.

Handles memory persistence, deduplication via embedding similarity,
corroboration, and full-text search index maintenance.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.index_payloads import (
    embedding_text_hash,
    memory_embedding_text,
)
from memforge.memory.lifecycle import allowed_search_statuses
from memforge.models import Memory
from memforge.retrieval.document_index import DocumentVectorIndex
from memforge.retrieval.embeddings import EmbeddingCache, embed_texts
from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID
from memforge.storage.adapters.protocols import KeywordSearch, RelationalStore, VectorStore

logger = logging.getLogger(__name__)

__all__ = ["MemoryStore"]

DEDUP_CANDIDATE_LIMIT = 10


def _dedup_access_scope() -> AccessScope:
    """The permissive single-datastore scope for dedup candidate selection.

    In Phase 0 this matches the historical where={"status":"active"} filter;
    the per-writer access narrowing activates in a later phase.
    """
    return AccessScope(
        user_id=LOCAL_DEV_USER_ID,
        open_projects=frozenset(),
        member_projects=frozenset(),
        include_private=False,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="project-first",
    )


def _memory_embedding_text(memory: Memory) -> str:
    """Build the text that gets embedded for a memory.

    Type prefix causes memories of the same type to cluster in embedding space.
    """
    return memory_embedding_text(memory)


class MemoryStore:
    """Synchronized memory storage across SQLite, ChromaDB, and FTS5.

    Every write operation updates all three stores to keep them consistent.
    SQLite is the source of truth. ChromaDB can be rebuilt from it.
    """

    def __init__(
        self,
        relational: RelationalStore,
        keyword: KeywordSearch,
        vector: VectorStore,
        embed_cfg: dict,
        dedup_threshold: float = 0.08,
        audit_logger: MemoryAuditLogger | None = None,
        document_index: DocumentVectorIndex | None = None,
    ) -> None:
        self.relational = relational
        self.keyword = keyword
        self.vector = vector
        self.document_index = document_index or DocumentVectorIndex(None)
        self.embed_cfg = embed_cfg
        self.dedup_threshold = dedup_threshold
        self._embedding_cache = EmbeddingCache()
        self.audit_logger = audit_logger

    @property
    def collection(self) -> Any:
        """The underlying memory vector collection (index-health and tests)."""
        return self.vector.collection

    @property
    def db(self) -> Any:
        """The bound Database, reached through the relational handle.

        Row writes and their co-transactional FTS writes live in Database
        methods, so the store delegates to them through this handle.
        """
        return self.relational._db

    # -------------------------------------------------------------------
    # Core: Deduplicate and Insert
    # -------------------------------------------------------------------

    async def deduplicate_and_insert(
        self,
        memory: Memory,
        doc_id: str,
        source_type: str,
        entity_ids: list[int] | None = None,
        excerpt: str | None = None,
        scope: AccessScope | None = None,
    ) -> str:
        """Check for near-duplicates, then insert or corroborate.

        Returns "inserted", "corroborated", or "skipped".
        """
        context = self._operation_context(doc_id=doc_id)
        # Embed the candidate memory
        embedding_text = _memory_embedding_text(memory)
        embedding = await self._embed(embedding_text)

        # Query the vector channel for near-duplicates.
        try:
            dedup_scope = scope or _dedup_access_scope()
            candidates = await self.vector.query(
                embedding, dedup_scope, None, DEDUP_CANDIDATE_LIMIT
            )
        except Exception as e:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=memory.id,
                doc_id=doc_id,
                error=str(e),
                payload={"index": "chroma", "operation": "dedup_query"},
            )
            logger.error("Vector dedup query failed for %s: %s", memory.id, e)
            raise

        # vector.query returns (id, score); the vector store owns the distance
        # math, so it decides whether a candidate is within the dedup threshold.
        for existing_id, score in candidates:
            if not self.vector.within_dedup_threshold(self.dedup_threshold, score):
                continue

            existing = await self.db.get_memory(existing_id)
            if not existing or existing.status != "active":
                await self._emit(
                    "stale_chroma_candidate_detected",
                    "skipped",
                    context=context,
                    memory_id=existing_id,
                    doc_id=doc_id,
                    reason="Chroma returned a missing or non-active memory during deduplication",
                    payload={
                        "candidate_memory_id": memory.id,
                        "score": score,
                        "db_status": existing.status if existing else "missing",
                    },
                )
                logger.warning(
                    "Ignoring stale Chroma dedup candidate %s for %s (status=%s)",
                    existing_id, memory.id, existing.status if existing else "missing",
                )
                continue

            # Near-duplicate found, corroborate instead of creating.
            await self.add_source_support(
                existing_id,
                doc_id,
                source_type,
                excerpt,
                support_kind="extracted",
                context=context,
            )
            logger.debug(
                "Memory corroborated: %s (score=%.4f, doc=%s)",
                existing_id, score, doc_id,
            )
            return "corroborated"

        # No duplicate, insert new memory
        await self._insert_full(
            memory,
            doc_id,
            source_type,
            entity_ids,
            excerpt,
            context,
        )
        await self._emit(
            "memory_insert_committed",
            "committed",
            context=context,
            memory_id=memory.id,
            doc_id=doc_id,
            support_kind="extracted",
            reason="new memory inserted after deduplication",
            payload={"content_hash": memory.content_hash, "memory_type": memory.memory_type},
        )
        return "inserted"

    async def insert_memory(
        self,
        memory: Memory,
        doc_id: str,
        source_type: str,
        entity_ids: list[int] | None = None,
        excerpt: str | None = None,
    ) -> None:
        """Insert a memory without deduplication.

        Used for quarantined challenger memories where near-duplicate matching
        would incorrectly corroborate the incumbent being challenged.
        """
        context = self._operation_context(doc_id=doc_id)
        await self._insert_full(
            memory,
            doc_id,
            source_type,
            entity_ids,
            excerpt,
            context,
        )
        await self._emit(
            "memory_insert_committed",
            "committed",
            context=context,
            memory_id=memory.id,
            doc_id=doc_id,
            support_kind="extracted",
            reason="memory inserted without deduplication",
            payload={"content_hash": memory.content_hash, "memory_type": memory.memory_type},
        )

    # -------------------------------------------------------------------
    # Insert (all stores)
    # -------------------------------------------------------------------

    async def _insert_full(
        self,
        memory: Memory,
        doc_id: str,
        source_type: str,
        entity_ids: list[int] | None,
        excerpt: str | None,
        context: AuditContext,
    ) -> None:
        """Insert memory into SQLite + FTS5 + ChromaDB + link entities and sources."""
        inserted = False
        chroma_upsert_started = False
        try:
            # 1. SQLite: memories table + FTS5
            await self.db.insert_memory(memory)
            inserted = True
            await self._emit(
                "fts_upsert_committed",
                "committed",
                context=context,
                memory_id=memory.id,
                doc_id=doc_id,
                payload={"operation": "memory_insert"},
            )

            # 2. SQLite: memory_sources (provenance)
            await self.db.add_memory_source(
                memory.id,
                doc_id,
                source_type,
                excerpt,
            )

            # 3. SQLite: memory_entities (entity links)
            if entity_ids:
                for entity_id in entity_ids:
                    await self.db.link_memory_entity(memory.id, entity_id)
            await self.db.rebuild_memory_fts(
                memory.id,
                search_visible_statuses=set(allowed_search_statuses()),
            )

            # 4. ChromaDB: vector embedding
            await self._emit(
                "chroma_upsert_attempted",
                "attempted",
                context=context,
                memory_id=memory.id,
                doc_id=doc_id,
                payload={"operation": "memory_insert"},
            )
            chroma_upsert_started = True
            indexed_text = await self._canonical_memory_embedding_text(memory)
            indexed_embedding = await self._embed(indexed_text)
            await self.vector.upsert(
                ids=[memory.id],
                embeddings=[indexed_embedding],
                metadatas=[{
                    "memory_type": memory.memory_type,
                    "source_doc_id": doc_id,
                    "space_or_project": memory.project_key or "",
                    "confidence": memory.confidence,
                    "status": memory.status,
                    "content_hash": memory.content_hash,
                    "embedding_text_hash": embedding_text_hash(indexed_text),
                }],
            )
            await self._emit(
                "chroma_upsert_committed",
                "committed",
                context=context,
                memory_id=memory.id,
                doc_id=doc_id,
                payload={"operation": "memory_insert"},
            )
        except Exception as e:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=memory.id,
                doc_id=doc_id,
                error=str(e),
                payload={"index": "chroma", "operation": "memory_insert"},
            )
            logger.error("Memory insert failed for %s: %s", memory.id, e)
            rollback_error: Exception | None = None
            if inserted:
                try:
                    await self.db.purge_memory(memory.id)
                    await self._emit(
                        "memory_insert_rolled_back",
                        "committed",
                        context=context,
                        memory_id=memory.id,
                        doc_id=doc_id,
                        reason="memory_insert_failed",
                    )
                except Exception as cleanup_exc:
                    rollback_error = cleanup_exc
            if chroma_upsert_started:
                try:
                    await self._restore_memory_vector_snapshot(
                        None,
                        memory_id=memory.id,
                        context=context,
                        label="memory_insert_rollback",
                    )
                except Exception as cleanup_exc:
                    rollback_error = cleanup_exc
            if rollback_error:
                raise rollback_error
            raise

        logger.debug("Memory inserted: %s (%s)", memory.id, memory.memory_type)

    # -------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------

    async def update_memory(
        self,
        memory_id: str,
        new_content: str,
        new_confidence: float | None = None,
        new_tags: list[str] | None = None,
    ) -> None:
        """Update a memory's content across all stores."""
        context = self._operation_context()
        previous = await self.db.get_memory(memory_id)
        previous_vector = await self._memory_vector_snapshot(memory_id)
        memory = None
        try:
            await self.db.update_memory_content(memory_id, new_content, new_confidence, new_tags)

            # Re-embed and update ChromaDB
            memory = await self.db.get_memory(memory_id)
            if memory:
                embedding_text = await self._canonical_memory_embedding_text(memory)
                embedding = await self._embed(embedding_text)
                await self._emit(
                    "chroma_upsert_attempted",
                    "attempted",
                    context=context,
                    memory_id=memory_id,
                    payload={"operation": "memory_update"},
                )
                await self.vector.upsert(
                    ids=[memory_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "memory_type": memory.memory_type,
                        "space_or_project": memory.project_key or "",
                        "confidence": memory.confidence,
                        "status": memory.status,
                        "content_hash": memory.content_hash,
                        "embedding_text_hash": embedding_text_hash(embedding_text),
                    }],
                )
                await self._emit(
                    "chroma_upsert_committed",
                    "committed",
                    context=context,
                    memory_id=memory_id,
                    payload={"operation": "memory_update"},
                )
        except Exception as e:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=memory_id,
                error=str(e),
                payload={"index": "chroma", "operation": "memory_update"},
            )
            logger.error("Memory update failed for %s: %s", memory_id, e)
            if previous:
                await self._restore_memory_row(previous)
                await self._restore_memory_vector_snapshot(
                    previous_vector,
                    memory_id=memory_id,
                    context=context,
                    label="memory_update_rollback",
                )
            raise
        await self._emit(
            "memory_update_committed",
            "committed",
            context=context,
            memory_id=memory_id,
            payload={"content_hash": memory.content_hash if memory else None},
        )

    # -------------------------------------------------------------------
    # Supersede
    # -------------------------------------------------------------------

    async def supersede_memory(
        self,
        old_memory_id: str,
        new_memory: Memory,
        doc_id: str,
        source_type: str,
        entity_ids: list[int] | None = None,
        excerpt: str | None = None,
        replacement_reason: str | None = None,
    ) -> None:
        """Supersede an old memory with a new one, updating all stores.

        The old memory is marked as superseded in SQLite and removed from
        ChromaDB. The new memory is inserted into all three stores (SQLite,
        FTS5, ChromaDB) and linked to entities and sources.
        """
        context = self._operation_context(doc_id=doc_id)
        old_snapshot = await self.db.get_memory(old_memory_id)
        old_vector = await self._memory_vector_snapshot(old_memory_id)
        new_vector = await self._memory_vector_snapshot(new_memory.id)
        new_chroma_upsert_started = False
        await self._emit(
            "memory_supersede_attempted",
            "attempted",
            context=context,
            memory_id=old_memory_id,
            candidate_id=new_memory.id,
            reason=replacement_reason,
        )
        try:
            await self.db.supersede_memory(
                old_memory_id,
                new_memory,
                replacement_reason=replacement_reason,
            )
            await self._remove_from_search_indexes(old_memory_id, label="superseded", context=context)

            await self.db.add_memory_source(
                new_memory.id,
                doc_id,
                source_type,
                excerpt,
            )
            if entity_ids:
                for entity_id in entity_ids:
                    await self.db.link_memory_entity(new_memory.id, entity_id)
            await self.db.rebuild_memory_fts(
                new_memory.id,
                search_visible_statuses=set(allowed_search_statuses()),
            )
            await self._emit(
                "fts_upsert_committed",
                "committed",
                context=context,
                memory_id=new_memory.id,
                doc_id=doc_id,
                payload={"operation": "memory_supersede_insert"},
            )

            embedding_text = await self._canonical_memory_embedding_text(new_memory)
            embedding = await self._embed(embedding_text)
            try:
                await self._emit(
                    "chroma_upsert_attempted",
                    "attempted",
                    context=context,
                    memory_id=new_memory.id,
                    doc_id=doc_id,
                    payload={"operation": "memory_supersede_insert"},
                )
                new_chroma_upsert_started = True
                await self.vector.upsert(
                    ids=[new_memory.id],
                    embeddings=[embedding],
                    metadatas=[{
                        "memory_type": new_memory.memory_type,
                        "source_doc_id": doc_id,
                        "space_or_project": new_memory.project_key or "",
                        "confidence": new_memory.confidence,
                        "status": new_memory.status,
                        "content_hash": new_memory.content_hash,
                        "embedding_text_hash": embedding_text_hash(embedding_text),
                    }],
                )
                await self._emit(
                    "chroma_upsert_committed",
                    "committed",
                    context=context,
                    memory_id=new_memory.id,
                    doc_id=doc_id,
                    payload={"operation": "memory_supersede_insert"},
                )
            except Exception as exc:
                await self._emit(
                    "index_operation_failed",
                    "failed",
                    context=context,
                    memory_id=new_memory.id,
                    doc_id=doc_id,
                    error=str(exc),
                    payload={"index": "chroma", "operation": "memory_supersede_insert"},
                )
                raise

            await self._emit(
                "memory_supersede_committed",
                "committed",
                context=context,
                memory_id=old_memory_id,
                candidate_id=new_memory.id,
                doc_id=doc_id,
                reason=replacement_reason,
                payload={"old_memory_id": old_memory_id, "new_memory_id": new_memory.id},
            )
        except Exception as exc:
            rollback_error: Exception | None = None
            try:
                await self.db.purge_memory(new_memory.id)
            except Exception as cleanup_exc:
                rollback_error = cleanup_exc
            if old_snapshot:
                try:
                    await self._restore_memory_row(old_snapshot)
                except Exception as cleanup_exc:
                    rollback_error = rollback_error or cleanup_exc
                try:
                    if old_vector:
                        await self._restore_memory_vector_snapshot(
                            old_vector,
                            memory_id=old_memory_id,
                            context=context,
                            label="supersede_rollback",
                        )
                    else:
                        await self._restore_search_indexes(
                            old_snapshot,
                            context=context,
                            label="supersede_rollback",
                        )
                except Exception as cleanup_exc:
                    rollback_error = rollback_error or cleanup_exc
            if new_chroma_upsert_started or new_vector:
                try:
                    await self._restore_memory_vector_snapshot(
                        new_vector,
                        memory_id=new_memory.id,
                        context=context,
                        label="supersede_rollback",
                    )
                except Exception as cleanup_exc:
                    rollback_error = rollback_error or cleanup_exc
            if rollback_error:
                raise rollback_error
            raise exc

        logger.info(
            "Memory superseded: %s -> %s (%s)",
            old_memory_id, new_memory.id, new_memory.memory_type,
        )

    # -------------------------------------------------------------------
    # Soft delete
    # -------------------------------------------------------------------

    async def soft_delete_memory(self, memory_id: str) -> None:
        """Compatibility wrapper for retiring a memory."""
        await self.retire_memory(memory_id, reason="admin_hidden")

    async def retire_memory(
        self,
        memory_id: str,
        reason: str = "admin_hidden",
        *,
        context: AuditContext | None = None,
        review_id: str | None = None,
    ) -> None:
        """Mark memory as retired and remove from search (ChromaDB + FTS5)."""
        context = context or self._operation_context()
        previous = await self.db.get_memory(memory_id)
        await self.db.update_memory_status(memory_id, "retired", reason=reason)
        try:
            await self._remove_from_search_indexes(memory_id, label="retired", context=context)
        except Exception:
            if previous:
                await self._restore_memory_row(previous)
                await self._restore_search_indexes(previous, context=context, label="retire_rollback")
            raise
        await self._emit(
            "memory_retire_committed",
            "committed",
            context=context,
            memory_id=memory_id,
            review_id=review_id,
            reason=reason,
        )

    async def retire_expired_memories(self) -> int:
        """Retire expired memories through the store boundary."""
        expired = await self.db.get_expired_memories()
        for memory in expired:
            await self.retire_memory(memory.id, reason="expired")
        return len(expired)

    async def add_source_support(
        self,
        memory_id: str,
        doc_id: str,
        source_type: str,
        excerpt: str | None = None,
        *,
        support_kind: str = "extracted",
        context: AuditContext | None = None,
    ) -> str:
        """Add or update source support for an existing memory."""
        context = context or self._operation_context(doc_id=doc_id)
        await self._emit(
            "source_support_add_attempted",
            "attempted",
            context=context,
            memory_id=memory_id,
            doc_id=doc_id,
            support_kind=support_kind,
        )
        outcome = await self.db.corroborate_memory(
            memory_id,
            doc_id,
            source_type,
            excerpt,
            support_kind=support_kind,
        )
        event_type = {
            "inserted": "source_support_added",
            "updated": "source_support_updated",
            "unchanged": "source_support_unchanged",
        }.get(outcome, "source_support_unchanged")
        await self._emit(
            event_type,
            "committed",
            context=context,
            memory_id=memory_id,
            doc_id=doc_id,
            support_kind=support_kind,
            payload={"outcome": outcome},
        )
        return outcome

    async def remove_source_support(
        self,
        memory_id: str,
        doc_id: str,
        reason: str = "no_support",
        *,
        context: AuditContext | None = None,
    ) -> bool:
        """Remove one source link and retire/hide the memory if support reaches zero."""
        context = context or self._operation_context(doc_id=doc_id)
        previous = await self.db.get_memory(memory_id)
        previous_sources = await self.db.get_memory_sources(memory_id)
        retired = await self.db.remove_memory_source(memory_id, doc_id, retire_reason=reason)
        if retired:
            try:
                await self._remove_from_search_indexes(memory_id, label="retired", context=context)
            except Exception:
                if previous:
                    await self._restore_memory_row(previous)
                    await self._restore_search_indexes(previous, context=context, label="source_support_rollback")
                for source in previous_sources:
                    await self.db.restore_memory_source_snapshot(source)
                raise
            await self._emit(
                "source_support_removal_retired_memory",
                "committed",
                context=context,
                memory_id=memory_id,
                doc_id=doc_id,
                reason=reason,
            )
        await self._emit(
            "source_support_removed",
            "committed",
            context=context,
            memory_id=memory_id,
            doc_id=doc_id,
            reason=reason,
            payload={"retired": retired},
        )
        return retired

    async def delete_document(
        self,
        doc_id: str,
        *,
        deletion_context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Delete a document and remove newly retired memories from search indexes."""
        context = self._operation_context(doc_id=doc_id)
        document_snapshot = await self.db.get_document(doc_id)
        document_side_snapshot = await self.db.get_document_side_table_snapshots([doc_id])
        document_vector_snapshot = self._document_vector_snapshot(doc_id)
        memory_ids = await self._memory_ids_for_doc(doc_id)
        memory_snapshots = await self._memory_snapshots(memory_ids)
        source_snapshots = await self._source_snapshots(memory_ids)
        try:
            await self._remove_document_vector(doc_id, context=context)
        except Exception:
            await self._restore_document_vector_snapshot(doc_id, document_vector_snapshot, context=context)
            raise
        try:
            retired_ids = await self.db.delete_document(doc_id)
        except Exception:
            await self._restore_deleted_document_state(
                document_snapshot=document_snapshot,
                document_side_snapshot=document_side_snapshot,
                document_vector_snapshot=document_vector_snapshot,
                memory_snapshots=memory_snapshots,
                source_snapshots=source_snapshots,
                context=context,
            )
            raise
        try:
            await self._remove_retired_from_search_indexes(retired_ids, context=context)
        except Exception:
            await self._restore_deleted_document_state(
                document_snapshot=document_snapshot,
                document_side_snapshot=document_side_snapshot,
                document_vector_snapshot=document_vector_snapshot,
                memory_snapshots=memory_snapshots,
                source_snapshots=source_snapshots,
                context=context,
            )
            raise
        await self._emit(
            "document_delete_committed",
            "committed",
            context=context,
            doc_id=doc_id,
            payload={
                **(deletion_context or {}),
                "retired_memory_ids": retired_ids,
            },
        )
        return retired_ids

    async def delete_source_cascade(self, source_id: str) -> list[str]:
        """Delete a source and remove newly retired memories from search indexes."""
        context = self._operation_context(source_id=source_id)
        source_snapshot = await self.db.get_source(source_id)
        document_snapshots = await self.db.list_documents(source=source_id, limit=100000)
        doc_ids = [doc.doc_id for doc in document_snapshots]
        document_side_snapshot = await self.db.get_document_side_table_snapshots(
            doc_ids,
            source_id=source_id,
        )
        document_vector_snapshots = {
            doc.doc_id: self._document_vector_snapshot(doc.doc_id)
            for doc in document_snapshots
        }
        memory_ids = await self._memory_ids_for_docs(doc_ids)
        memory_snapshots = await self._memory_snapshots(memory_ids)
        source_snapshots = await self._source_snapshots(memory_ids)
        try:
            for doc in document_snapshots:
                await self._remove_document_vector(doc.doc_id, context=context)
        except Exception:
            for doc in document_snapshots:
                await self._restore_document_vector_snapshot(
                    doc.doc_id,
                    document_vector_snapshots.get(doc.doc_id),
                    context=context,
                )
            raise
        try:
            retired_ids = await self.db.delete_source_cascade(source_id)
        except Exception:
            await self._restore_deleted_source_state(
                source_snapshot=source_snapshot,
                document_snapshots=document_snapshots,
                document_side_snapshot=document_side_snapshot,
                document_vector_snapshots=document_vector_snapshots,
                memory_snapshots=memory_snapshots,
                source_snapshots=source_snapshots,
                context=context,
            )
            raise
        try:
            await self._remove_retired_from_search_indexes(retired_ids, context=context)
        except Exception:
            await self._restore_deleted_source_state(
                source_snapshot=source_snapshot,
                document_snapshots=document_snapshots,
                document_side_snapshot=document_side_snapshot,
                document_vector_snapshots=document_vector_snapshots,
                memory_snapshots=memory_snapshots,
                source_snapshots=source_snapshots,
                context=context,
            )
            raise
        await self._emit(
            "source_delete_cascade_committed",
            "committed",
            context=context,
            source_id=source_id,
            payload={"retired_memory_ids": retired_ids},
        )
        return retired_ids

    async def mark_pending_review(self, memory_id: str, reason: str | None = None) -> None:
        """Quarantine a memory until a human or future workflow resolves it."""
        context = self._operation_context()
        previous = await self.db.get_memory(memory_id)
        await self.db.update_memory_status(memory_id, "pending_review", reason=reason)
        try:
            await self._remove_from_search_indexes(memory_id, label="pending_review", context=context)
        except Exception:
            if previous:
                await self._restore_memory_row(previous)
                await self._restore_search_indexes(previous, context=context, label="pending_review_rollback")
            raise
        await self._emit(
            "memory_pending_review_committed",
            "committed",
            context=context,
            memory_id=memory_id,
            reason=reason,
        )

    async def purge_memory(self, memory_id: str) -> bool:
        """Hard-delete a memory from SQLite, FTS5, and ChromaDB."""
        context = self._operation_context()
        previous = await self.db.get_memory(memory_id)
        previous_vector = await self._memory_vector_snapshot(memory_id)
        await self._emit(
            "memory_purge_attempted",
            "attempted",
            context=context,
            memory_id=memory_id,
        )
        try:
            await self._emit(
                "chroma_delete_attempted",
                "attempted",
                context=context,
                memory_id=memory_id,
                payload={"label": "purged"},
            )
            await self.vector.delete([memory_id])
            await self._emit(
                "chroma_delete_committed",
                "committed",
                context=context,
                memory_id=memory_id,
                payload={"label": "purged"},
            )
        except Exception as e:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=memory_id,
                error=str(e),
                payload={"index": "chroma", "operation": "delete", "label": "purged"},
            )
            logger.warning("ChromaDB delete failed for purged %s: %s", memory_id, e)
            if previous:
                await self._restore_memory_vector_snapshot(
                    previous_vector,
                    memory_id=memory_id,
                    context=context,
                    label="purge_rollback",
                )
            raise
        try:
            purged = await self.db.purge_memory(memory_id)
        except Exception:
            if previous:
                await self._restore_search_indexes(previous, context=context, label="purge_rollback")
            raise
        await self._emit(
            "memory_purge_committed",
            "committed",
            context=context,
            memory_id=memory_id,
            payload={"purged": purged},
        )
        if purged:
            await self.db.redact_memory_audit_payloads(memory_id)
        return purged

    async def promote_quarantined_challenger(
        self,
        *,
        incumbent: Memory,
        challenger: Memory,
        replacement_reason: str | None = None,
        review_id: str | None = None,
        context: AuditContext | None = None,
    ) -> None:
        """Promote a reviewed challenger and retire the incumbent from search."""
        context = context or self._operation_context()
        incumbent_snapshot = await self.db.get_memory(incumbent.id)
        challenger_snapshot = await self.db.get_memory(challenger.id)
        incumbent_vector = await self._memory_vector_snapshot(incumbent.id)
        challenger_vector = await self._memory_vector_snapshot(challenger.id)
        promotion_phase = "db_promote"
        try:
            await self.db.promote_quarantined_challenger(
                incumbent_id=incumbent.id,
                challenger=challenger,
                replacement_reason=replacement_reason,
            )
            promotion_phase = "fts_upsert"
            await self._emit(
                "fts_upsert_committed",
                "committed",
                context=context,
                memory_id=challenger.id,
                review_id=review_id,
                payload={"operation": "review_promote_challenger"},
            )
            promotion_phase = "incumbent_index_delete"
            await self._remove_from_search_indexes(incumbent.id, label="superseded", context=context)
            promotion_phase = "challenger_chroma_upsert"
            await self._emit(
                "chroma_upsert_attempted",
                "attempted",
                context=context,
                memory_id=challenger.id,
                review_id=review_id,
                payload={"operation": "review_promote_challenger"},
            )
            embedding_text = await self._canonical_memory_embedding_text(challenger)
            embedding = await self._embed(embedding_text)
            await self.vector.upsert(
                ids=[challenger.id],
                embeddings=[embedding],
                metadatas=[{
                    "memory_type": challenger.memory_type,
                    "space_or_project": challenger.project_key or "",
                    "confidence": challenger.confidence,
                    "status": "active",
                    "content_hash": challenger.content_hash,
                    "embedding_text_hash": embedding_text_hash(embedding_text),
                }],
            )
            await self._emit(
                "chroma_upsert_committed",
                "committed",
                context=context,
                memory_id=challenger.id,
                review_id=review_id,
                payload={"operation": "review_promote_challenger"},
            )
        except Exception as e:
            if promotion_phase == "challenger_chroma_upsert":
                await self._emit(
                    "index_operation_failed",
                    "failed",
                    context=context,
                    memory_id=challenger.id,
                    review_id=review_id,
                    error=str(e),
                    payload={"index": "chroma", "operation": "review_promote_challenger"},
                )
            logger.error("Review promotion failed during %s for %s: %s", promotion_phase, challenger.id, e)
            rollback_error: Exception | None = None
            if incumbent_snapshot:
                await self._restore_memory_row(incumbent_snapshot)
            if challenger_snapshot:
                await self._restore_memory_row(challenger_snapshot)
            try:
                await self._restore_memory_vector_snapshot(
                    incumbent_vector,
                    memory_id=incumbent.id,
                    context=context,
                    label="review_promote_rollback",
                )
            except Exception as rollback_exc:
                rollback_error = rollback_exc
            try:
                await self._restore_memory_vector_snapshot(
                    challenger_vector,
                    memory_id=challenger.id,
                    context=context,
                    label="review_promote_rollback",
                )
            except Exception as rollback_exc:
                rollback_error = rollback_error or rollback_exc
            if rollback_error:
                logger.error("Review promotion rollback had an index restore error: %s", rollback_error)
            raise
        await self._emit(
            "memory_supersede_committed",
            "committed",
            context=context,
            memory_id=incumbent.id,
            candidate_id=challenger.id,
            review_id=review_id,
            reason=replacement_reason,
            payload={"old_memory_id": incumbent.id, "new_memory_id": challenger.id},
        )

    async def record_review_decision(
        self,
        event_type: str,
        *,
        memory_id: str,
        review_id: str,
        reviewer: str | None,
        reason: str | None = None,
        context: AuditContext | None = None,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Record a resolved human review decision after the review row commits."""
        await self._emit(
            event_type,
            "failed" if error else "committed",
            context=context or self._operation_context(),
            actor_id=reviewer,
            memory_id=memory_id,
            review_id=review_id,
            reason=reason,
            payload=payload or {},
            error=error,
        )

    async def record_audit_event(
        self,
        event_type: str,
        status: str,
        *,
        context: AuditContext | None = None,
        **fields: Any,
    ) -> None:
        """Record a memory audit event for pipeline decisions that do not mutate storage."""
        await self._emit(event_type, status, context=context or self._operation_context(), **fields)

    async def restore_review_transition(
        self,
        *,
        incumbent: Memory | None,
        challenger: Memory,
        context: AuditContext,
        review_id: str,
        reason: str,
    ) -> None:
        """Restore review participant memory state after review-row resolution fails."""
        if incumbent:
            await self._restore_memory_row(incumbent)
            await self._restore_search_indexes(incumbent, context=context, label=reason)
        await self._restore_memory_row(challenger)
        if challenger.status not in set(allowed_search_statuses()):
            await self._remove_from_search_indexes(challenger.id, label=reason, context=context)
        await self._emit(
            "review_memory_transition_rolled_back",
            "committed",
            context=context,
            memory_id=challenger.id,
            review_id=review_id,
            reason=reason,
        )

    async def _remove_retired_from_search_indexes(
        self,
        memory_ids: list[str],
        *,
        context: AuditContext,
    ) -> None:
        for memory_id in dict.fromkeys(memory_ids):
            await self._remove_from_search_indexes(memory_id, label="retired", context=context)

    async def _remove_from_search_indexes(
        self,
        memory_id: str,
        label: str,
        *,
        context: AuditContext,
    ) -> None:
        """Remove a non-active memory from FTS5 and ChromaDB."""
        # Remove from FTS5 so it doesn't appear in keyword searches
        try:
            await self._emit(
                "fts_delete_attempted",
                "attempted",
                context=context,
                memory_id=memory_id,
                payload={"label": label},
            )
            await self.keyword.remove(memory_id)
            await self._emit(
                "fts_delete_committed",
                "committed",
                context=context,
                memory_id=memory_id,
                payload={"label": label},
            )
        except Exception as e:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=memory_id,
                error=str(e),
                payload={"index": "fts5", "operation": "delete", "label": label},
            )
            logger.warning("FTS5 delete failed for %s %s: %s", label, memory_id, e)
            raise
        # Remove from ChromaDB so it doesn't appear in vector searches
        try:
            await self._emit(
                "chroma_delete_attempted",
                "attempted",
                context=context,
                memory_id=memory_id,
                payload={"label": label},
            )
            await self.vector.delete([memory_id])
            await self._emit(
                "chroma_delete_committed",
                "committed",
                context=context,
                memory_id=memory_id,
                payload={"label": label},
            )
        except Exception as e:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=memory_id,
                error=str(e),
                payload={"index": "chroma", "operation": "delete", "label": label},
            )
            logger.warning("ChromaDB delete failed for %s: %s", memory_id, e)
            raise

    async def _remove_document_vector(self, doc_id: str, *, context: AuditContext) -> None:
        if not self.document_index.enabled:
            return
        try:
            await self._emit(
                "document_chroma_delete_attempted",
                "attempted",
                context=context,
                doc_id=doc_id,
                payload={"index": "document_chroma"},
            )
            self.document_index.delete(doc_id)
            await self._emit(
                "document_chroma_delete_committed",
                "committed",
                context=context,
                doc_id=doc_id,
                payload={"index": "document_chroma"},
            )
        except Exception as e:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                doc_id=doc_id,
                error=str(e),
                payload={"index": "document_chroma", "operation": "delete"},
            )
            raise

    async def _restore_memory_row(self, memory: Memory) -> None:
        """Restore SQLite memory/FTS state after an external index write fails."""
        await self.db.restore_memory_snapshot(
            memory,
            search_visible_statuses=set(allowed_search_statuses()),
        )

    async def _restore_search_indexes(
        self,
        memory: Memory,
        *,
        context: AuditContext,
        label: str,
    ) -> None:
        if memory.status not in set(allowed_search_statuses()):
            return
        try:
            await self._emit(
                "chroma_upsert_attempted",
                "attempted",
                context=context,
                memory_id=memory.id,
                payload={"operation": label},
            )
            embedding_text = await self._canonical_memory_embedding_text(memory)
            embedding = await self._embed(embedding_text)
            await self.vector.upsert(
                ids=[memory.id],
                embeddings=[embedding],
                metadatas=[{
                    "memory_type": memory.memory_type,
                    "space_or_project": memory.project_key or "",
                    "confidence": memory.confidence,
                    "status": memory.status,
                    "content_hash": memory.content_hash,
                    "embedding_text_hash": embedding_text_hash(embedding_text),
                }],
            )
            await self._emit(
                "chroma_upsert_committed",
                "committed",
                context=context,
                memory_id=memory.id,
                payload={"operation": label},
            )
            await self._emit(
                "memory_mutation_rolled_back",
                "committed",
                context=context,
                memory_id=memory.id,
                reason=label,
                payload={"restored_status": memory.status},
            )
        except Exception as exc:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=memory.id,
                error=str(exc),
                payload={"index": "chroma", "operation": label},
            )
            raise

    async def _memory_ids_for_doc(self, doc_id: str) -> list[str]:
        """Document-cascade row lookup. Reads provenance rows through the bound
        connection: a row helper, not a retrieval channel, so it stays outside
        the adapters in this phase."""
        ids: list[str] = []
        async with self.db.db.execute(
            "SELECT memory_id FROM memory_sources WHERE doc_id = ?",
            (doc_id,),
        ) as cursor:
            async for row in cursor:
                ids.append(row[0])
        return list(dict.fromkeys(ids))

    async def _memory_ids_for_docs(self, doc_ids: list[str]) -> list[str]:
        ids: list[str] = []
        for doc_id in doc_ids:
            ids.extend(await self._memory_ids_for_doc(doc_id))
        return list(dict.fromkeys(ids))

    async def _memory_snapshots(self, memory_ids: list[str]) -> list[Memory]:
        snapshots: list[Memory] = []
        for memory_id in dict.fromkeys(memory_ids):
            memory = await self.db.get_memory(memory_id)
            if memory:
                snapshots.append(memory)
        return snapshots

    async def _source_snapshots(self, memory_ids: list[str]):
        snapshots = []
        for memory_id in dict.fromkeys(memory_ids):
            snapshots.extend(await self.db.get_memory_sources(memory_id))
        return snapshots

    def _document_vector_snapshot(self, doc_id: str) -> dict[str, Any] | None:
        return self.document_index.snapshot(doc_id)

    async def _memory_vector_snapshot(self, memory_id: str) -> dict[str, Any] | None:
        record = await self.vector.get_record(memory_id)
        if record is None:
            return None
        return {
            "id": record["id"],
            "embedding": record.get("embedding"),
            "document": None,
            "metadata": record.get("metadata") or {},
        }

    async def _restore_deleted_document_state(
        self,
        *,
        document_snapshot,
        document_side_snapshot,
        document_vector_snapshot: dict[str, Any] | None,
        memory_snapshots: list[Memory],
        source_snapshots,
        context: AuditContext,
    ) -> None:
        if document_snapshot:
            await self.db.restore_document_snapshot(document_snapshot)
            await self.db.restore_document_side_table_snapshots(document_side_snapshot)
            await self._restore_document_vector_snapshot(
                document_snapshot.doc_id,
                document_vector_snapshot,
                context=context,
            )
        for memory in memory_snapshots:
            await self._restore_memory_row(memory)
            await self._restore_search_indexes(memory, context=context, label="document_delete_rollback")
        for source in source_snapshots:
            await self.db.restore_memory_source_snapshot(source)

    async def _restore_deleted_source_state(
        self,
        *,
        source_snapshot,
        document_snapshots,
        document_side_snapshot,
        document_vector_snapshots: dict[str, dict[str, Any] | None],
        memory_snapshots: list[Memory],
        source_snapshots,
        context: AuditContext,
    ) -> None:
        if source_snapshot:
            await self.db.restore_source_snapshot(source_snapshot)
        for document in document_snapshots:
            await self.db.restore_document_snapshot(document)
        await self.db.restore_document_side_table_snapshots(document_side_snapshot)
        for document in document_snapshots:
            await self._restore_document_vector_snapshot(
                document.doc_id,
                document_vector_snapshots.get(document.doc_id),
                context=context,
            )
        for memory in memory_snapshots:
            await self._restore_memory_row(memory)
            await self._restore_search_indexes(memory, context=context, label="source_delete_rollback")
        for source in source_snapshots:
            await self.db.restore_memory_source_snapshot(source)

    async def _restore_document_vector_snapshot(
        self,
        doc_id: str,
        snapshot: dict[str, Any] | None,
        *,
        context: AuditContext,
    ) -> None:
        if not self.document_index.enabled:
            return
        try:
            self.document_index.restore(doc_id, snapshot)
        except Exception as exc:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                doc_id=doc_id,
                error=str(exc),
                payload={"index": "document_chroma", "operation": "restore"},
            )
            raise

    async def _restore_memory_vector_snapshot(
        self,
        snapshot: dict[str, Any] | None,
        *,
        memory_id: str,
        context: AuditContext,
        label: str,
    ) -> None:
        if snapshot is None:
            try:
                await self._emit(
                    "chroma_delete_attempted",
                    "attempted",
                    context=context,
                    memory_id=memory_id,
                    payload={"label": label},
                )
                await self.vector.delete([memory_id])
                await self._emit(
                    "chroma_delete_committed",
                    "committed",
                    context=context,
                    memory_id=memory_id,
                    payload={"label": label},
                )
            except Exception as exc:
                await self._emit(
                    "index_operation_failed",
                    "failed",
                    context=context,
                    memory_id=memory_id,
                    error=str(exc),
                    payload={"index": "chroma", "operation": label},
                )
                raise
            return

        embedding = snapshot.get("embedding")
        # A snapshot with no stored embedding cannot be re-upserted, so the
        # vector record is dropped instead, matching the None-snapshot branch.
        if embedding is None:
            await self.vector.delete([memory_id])
            return
        try:
            await self._emit(
                "chroma_upsert_attempted",
                "attempted",
                context=context,
                memory_id=memory_id,
                payload={"operation": label},
            )
            await self.vector.upsert(
                ids=[snapshot["id"]],
                embeddings=[embedding],
                metadatas=[snapshot.get("metadata") or {}],
            )
            await self._emit(
                "chroma_upsert_committed",
                "committed",
                context=context,
                memory_id=memory_id,
                payload={"operation": label},
            )
        except Exception as exc:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=memory_id,
                error=str(exc),
                payload={"index": "chroma", "operation": label},
            )
            raise

    async def _emit(
        self,
        event_type: str,
        status: str,
        *,
        context: AuditContext | None = None,
        **fields: Any,
    ) -> None:
        if not self.audit_logger:
            return
        await self.audit_logger.emit(event_type, status, context=context, **fields)

    def _operation_context(self, **fields: str | None) -> AuditContext:
        base = self.audit_logger.default_context if self.audit_logger else AuditContext()
        return base.child(**fields)

    def operation_context(self, **fields: str | None) -> AuditContext:
        """Create a context shared by multiple public store operations."""
        return self._operation_context(**fields)

    async def _canonical_memory_embedding_text(self, memory: Memory) -> str:
        entity_names = await self.db.get_memory_entity_names(memory.id)
        return memory_embedding_text(memory, entity_names if entity_names else None)

    # -------------------------------------------------------------------
    # Embedding helper
    # -------------------------------------------------------------------

    async def _embed(self, text: str) -> list[float]:
        """Embed text, with LRU cache for repeated queries."""
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached

        vectors = await asyncio.to_thread(
            embed_texts,
            [text],
            self.embed_cfg["base_url"],
            self.embed_cfg["api_key"],
            self.embed_cfg["model"],
        )
        embedding = vectors[0]
        self._embedding_cache.put(text, embedding)
        return embedding
