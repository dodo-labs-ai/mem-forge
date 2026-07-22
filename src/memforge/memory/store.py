"""Memory Store - SQLite + ChromaDB + FTS5 synchronized storage.

Handles memory persistence, deduplication via embedding similarity,
corroboration, and full-text search index maintenance.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from hashlib import sha256
from itertools import zip_longest
from typing import Any, Sequence

from memforge.memory.audit import AuditContext, MemoryAuditLogger
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
    relation_run_id_for,
)
from memforge.memory.index_payloads import (
    embedding_text_hash,
    memory_embedding_text,
)
from memforge.memory.lifecycle import allowed_search_statuses
from memforge.memory.lifecycle_plan import (
    LifecyclePlan,
    LifecycleVectorDeliveryResult,
    LifecycleVectorDeliveryState,
    LifecycleVectorOperation,
)
from memforge.memory.lifecycle_planner import lifecycle_access_context_hash
from memforge.models import (
    Memory,
    MemoryReview,
    MemoryStatus,
    ReplacementKind,
    UNSORTED_PROJECT_KEY,
    VIRTUAL_DOCUMENT_SOURCE_IDS,
    Visibility,
)
from memforge.retrieval.embeddings import EmbeddingCache, embed_texts
from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID
from memforge.storage.adapters.protocols import KeywordSearch, RelationalStore, VectorStore
from memforge.source_activity import SourceActivityLease
from memforge.source_projection import SourceProjection

logger = logging.getLogger(__name__)

__all__ = ["MemoryStore"]

DEDUP_CANDIDATE_LIMIT = 10
AGENT_CLAIM_RECONCILE_LIMIT = 10
AGENT_CLAIM_RECONCILE_SIMILARITY_FLOOR = 0.85
LIFECYCLE_VECTOR_DELIVERY_BATCH_SIZE = 100


def _writer_access_scope(memory: Memory) -> AccessScope:
    """The dedup scope a writer of this memory must use.

    A private writer asks under its own user id; a workspace writer asks
    under the local dev principal. Cross-visibility merges are blocked by
    the access predicate plus the visibility-mismatch guard in
    deduplicate_and_insert.
    """
    if memory.visibility == Visibility.PRIVATE.value:
        return AccessScope(
            user_id=memory.owner_user_id or LOCAL_DEV_USER_ID,
            include_private=True,
            allowed_statuses=(MemoryStatus.ACTIVE.value,),
            active_project=memory.project_key,
            scope_mode="project-first",
        )
    return AccessScope(
        user_id=LOCAL_DEV_USER_ID,
        include_private=False,
        allowed_statuses=(MemoryStatus.ACTIVE.value,),
        active_project=memory.project_key,
        scope_mode="project-first",
    )


def _memory_embedding_text(memory: Memory) -> str:
    """Build the text that gets embedded for a memory.

    Type prefix causes memories of the same type to cluster in embedding space.
    """
    return memory_embedding_text(memory)


def _parse_iso_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _dedup_support_evidence_unit_id(
    *,
    source_id: str,
    doc_id: str,
    doc_revision_id: str | None,
    candidate_memory_id: str,
    target_memory_id: str,
    content_hash: str,
) -> str:
    digest = sha256(
        "\x1f".join(
            [
                source_id,
                doc_id,
                doc_revision_id or "",
                candidate_memory_id,
                target_memory_id,
                content_hash,
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"eu-dedup-support-{digest}"


def _dedup_support_relation_run_id(unit: EvidenceUnit, target_memory_id: str) -> str:
    return relation_run_id_for(
        prefix="dedup-support",
        unit=unit,
        action=LifecycleAction.ATTACH_SUPPORT,
        classifier_version="dedup-support-v1",
        candidate_memory_id=target_memory_id,
        relation_type=RelationType.SUPPORTS,
        authority_case=AuthorityCase.INDEPENDENT_SUPPORT,
        bucket=CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
    )


def _memory_metadata(
    memory: Memory,
    *,
    embedding_text_hash: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Vector metadata payload for a live memory write.

    Single source of truth for the keys stamped on every memory upsert, so
    every write site agrees with the access predicate's pre-filter and the
    repair pass's rewritten payload.
    """
    base: dict[str, Any] = {
        "memory_type": memory.memory_type,
        "project_key": memory.project_key or "",
        "repo_identifier": memory.repo_identifier or "",
        "visibility": memory.visibility,
        "owner_user_id": memory.owner_user_id or "",
        "confidence": memory.confidence,
        "status": memory.status,
        "content_hash": memory.content_hash,
        "embedding_text_hash": embedding_text_hash,
    }
    if extra:
        base.update(extra)
    return base


def _normalize_snapshot_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Carry a saved snapshot's vector metadata across the rename and the new
    visibility/owner keys. A pre-rename snapshot has the legacy
    `space_or_project` key and lacks `visibility`/`owner_user_id`; a snapshot
    taken after has the new keys. This helper makes either replayable."""
    out = dict(metadata or {})
    if "project_key" not in out and "space_or_project" in out:
        out["project_key"] = out.pop("space_or_project")
    out.pop("space_or_project", None)
    out.setdefault("visibility", Visibility.WORKSPACE.value)
    out.setdefault("owner_user_id", "")
    out.setdefault("project_key", "")
    out.setdefault("repo_identifier", "")
    return out


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
    ) -> None:
        self.relational = relational
        self.keyword = keyword
        self.vector = vector
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

    async def rebucket_project_memories(
        self,
        affected_ids: Sequence[str],
        new_project_key: str,
    ) -> None:
        """Rewrite the vector metadata `project_key` for the affected
        memories so the embedding channel agrees with the relational
        rebucket about where each row now lives.

        Project deletion runs this first, then commits the relational
        rebucket. Driving the vector update first means a failure here
        leaves both stores still pointing at the original project, with
        no half-applied state to reconcile. The predicate would otherwise
        find rows under the new key while the vector channel kept
        scoring them under the previous one.

        If a per-id upsert fails partway through the batch, every record
        already moved is restored to its pre-call metadata before the
        original exception is re-raised. The relational rebucket has not
        run yet, so a clean rollback here returns the system to the
        exact state the caller observed before invoking this method.
        """
        if not affected_ids:
            return
        # Snapshot each record before any mutation so a partial-failure
        # rollback can restore exactly what the vector channel held
        # before this call. Records the vector store does not know about
        # (returned None) or that lack an embedding are ignored, the
        # same way the original mutation skips them.
        snapshots: list[tuple[str, list[float], dict[str, Any]]] = []
        for memory_id in affected_ids:
            record = await self.vector.get_record(memory_id)
            if record is None:
                continue
            embedding = record.get("embedding")
            if embedding is None:
                continue
            metadata = dict(record.get("metadata") or {})
            snapshots.append((memory_id, embedding, metadata))

        applied: list[tuple[str, list[float], dict[str, Any]]] = []
        try:
            for memory_id, embedding, original_metadata in snapshots:
                rebucketed_metadata = dict(original_metadata)
                rebucketed_metadata["project_key"] = new_project_key
                await self.vector.upsert(
                    ids=[memory_id],
                    embeddings=[embedding],
                    metadatas=[rebucketed_metadata],
                )
                applied.append((memory_id, embedding, original_metadata))
        except Exception:
            # Roll back every record this call already moved so the
            # vector channel matches the relational state the caller
            # held before invoking us. A rollback failure is logged but
            # does not mask the original error.
            for memory_id, embedding, original_metadata in reversed(applied):
                try:
                    await self.vector.upsert(
                        ids=[memory_id],
                        embeddings=[embedding],
                        metadatas=[original_metadata],
                    )
                except Exception as rollback_err:  # pragma: no cover - defensive
                    logger.error(
                        "vector rollback failed for memory_id=%s: %s",
                        memory_id,
                        rollback_err,
                    )
            raise

    async def attempt_lifecycle_vector_delivery(
        self,
        lifecycle_plan_id: str | None = None,
        *,
        source_id: str | None = None,
    ) -> LifecycleVectorDeliveryResult:
        """Attempt durable vector work without changing relational commit semantics.

        Lifecycle plans commit their relational graph before this method runs.
        A vector or embedding failure therefore remains a durable outbox concern:
        callers receive ``PENDING`` and must not compensate the committed graph.
        """

        scope = lifecycle_plan_id or source_id or "all"
        try:
            tasks = await self.relational.list_lifecycle_vector_tasks(
                source_id=source_id,
                lifecycle_plan_id=lifecycle_plan_id,
                limit=LIFECYCLE_VECTOR_DELIVERY_BATCH_SIZE + 1,
            )
        except Exception as exc:
            logger.warning(
                "Lifecycle vector delivery lookup remains pending scope=%s error_type=%s",
                scope,
                type(exc).__name__,
                exc_info=True,
            )
            return LifecycleVectorDeliveryResult(
                state=LifecycleVectorDeliveryState.PENDING,
                error_types=(type(exc).__name__,),
            )

        more_work = len(tasks) > LIFECYCLE_VECTOR_DELIVERY_BATCH_SIZE
        selected_tasks = tasks[:LIFECYCLE_VECTOR_DELIVERY_BATCH_SIZE]
        delivered_tasks = 0
        error_types: list[str] = []
        for task in selected_tasks:
            try:
                if task.operation is LifecycleVectorOperation.DELETE:
                    await self.vector.delete([task.memory_id])
                else:
                    memory = await self.relational.get_memory(task.memory_id)
                    if memory is None or memory.status != MemoryStatus.ACTIVE.value:
                        raise ValueError(f"active lifecycle Memory missing: {task.memory_id}")
                    indexed_text = await self._canonical_memory_embedding_text(memory)
                    indexed_embedding = await self._embed(indexed_text)
                    sources = await self.relational.get_memory_sources(memory.id)
                    await self.vector.upsert(
                        ids=[memory.id],
                        embeddings=[indexed_embedding],
                        metadatas=[
                            _memory_metadata(
                                memory,
                                embedding_text_hash=embedding_text_hash(indexed_text),
                                extra={
                                    "source_doc_id": sources[0].doc_id if sources else "",
                                },
                            )
                        ],
                    )
                await self.relational.complete_lifecycle_vector_task(task.id)
                delivered_tasks += 1
            except Exception as exc:
                error_types.append(type(exc).__name__)
                try:
                    await self.relational.fail_lifecycle_vector_task(task.id, str(exc))
                except Exception as persistence_exc:
                    error_types.append(type(persistence_exc).__name__)
                    logger.warning(
                        "Lifecycle vector task failure could not be persisted task=%s scope=%s "
                        "error_type=%s persistence_error_type=%s",
                        task.id,
                        scope,
                        type(exc).__name__,
                        type(persistence_exc).__name__,
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        "Lifecycle vector task remains pending task=%s scope=%s error_type=%s",
                        task.id,
                        scope,
                        type(exc).__name__,
                    )

        failed_tasks = len(selected_tasks) - delivered_tasks
        if failed_tasks and not delivered_tasks:
            try:
                remaining_tasks = await self.relational.list_lifecycle_vector_tasks(
                    source_id=source_id,
                    lifecycle_plan_id=lifecycle_plan_id,
                    limit=1,
                )
            except Exception as exc:
                error_types.append(type(exc).__name__)
            else:
                if not remaining_tasks:
                    # Another source-scoped consumer may have completed the
                    # durable task after this batch listed it. Vector
                    # operations are idempotent, so an empty durable remainder
                    # is successful delivery rather than a false failure.
                    return LifecycleVectorDeliveryResult(
                        state=LifecycleVectorDeliveryState.DELIVERED,
                        attempted_tasks=len(selected_tasks),
                    )
        state = (
            LifecycleVectorDeliveryState.PENDING
            if failed_tasks or more_work
            else LifecycleVectorDeliveryState.DELIVERED
        )
        return LifecycleVectorDeliveryResult(
            state=state,
            attempted_tasks=len(selected_tasks),
            delivered_tasks=delivered_tasks,
            failed_tasks=failed_tasks,
            error_types=tuple(dict.fromkeys(error_types)),
        )

    # -------------------------------------------------------------------
    # Core: Deduplicate and Insert
    # -------------------------------------------------------------------

    async def find_agent_claim_memory_candidates(
        self,
        memory: Memory,
        *,
        source_id: str,
        owner_user_id: str,
        repo_identifier: str | None,
        limit: int = AGENT_CLAIM_RECONCILE_LIMIT,
    ) -> list[tuple[Memory, float]]:
        """Return active private same-user same-repo agent-session memory candidates."""
        embedding_text = await self._canonical_memory_embedding_text(memory)
        embedding = await self._embed(embedding_text)
        scope = AccessScope(
            user_id=owner_user_id,
            include_private=True,
            allowed_statuses=(MemoryStatus.ACTIVE.value,),
            active_project=memory.project_key,
            scope_mode="project-first",
            active_repo_identifier=repo_identifier,
        )
        hits = await self.vector.query(embedding, scope, None, limit)
        candidates: list[tuple[Memory, float]] = []
        for memory_id, score in hits:
            if score < AGENT_CLAIM_RECONCILE_SIMILARITY_FLOOR:
                continue
            existing = await self.db.get_memory(memory_id)
            if (
                existing is None
                or existing.status != MemoryStatus.ACTIVE.value
                or existing.visibility != Visibility.PRIVATE.value
                or existing.owner_user_id != owner_user_id
                or (existing.repo_identifier or None) != (repo_identifier or None)
            ):
                continue
            sources = await self.db.get_memory_sources(existing.id)
            if not any(
                source.source_type == "agent_session" and source.source_id == source_id
                for source in sources
            ):
                continue
            candidates.append((existing, score))
        return candidates

    async def deduplicate_and_insert(
        self,
        memory: Memory,
        doc_id: str,
        source_type: str,
        *,
        source_updated_at: datetime | None,
        entity_ids: list[int] | None = None,
        excerpt: str | None = None,
        scope: AccessScope | None = None,
        relation_outcome: RelationOutcomeBundle | None = None,
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
            dedup_scope = scope or _writer_access_scope(memory)
            candidates = await self.vector.query(embedding, dedup_scope, None, DEDUP_CANDIDATE_LIMIT)
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
                    existing_id,
                    memory.id,
                    existing.status if existing else "missing",
                )
                continue

            # The predicate decides what the writer can SEE; dedup is a write-side
            # decision that adds two narrower rules on top of "visible candidates":
            #   1. Same visibility tier: a private write must not corroborate a team row,
            #      and vice versa, even when the predicate exposes both.
            #   2. Same project scope:
            #      - workspace candidates must share project_key with the writer
            #        (vector channel does not pre-filter by project; cross-project
            #        merges would otherwise leak across project boundaries).
            #      - private candidates must share owner_user_id with the writer
            #        (private dedups against the same user's set only).
            if existing.visibility != memory.visibility:
                continue
            if memory.visibility == Visibility.WORKSPACE.value:
                # NULL project_key is normalized to UNSORTED at persistence time,
                # so apply the same normalization on both sides of the comparison
                # to keep same-project candidates eligible for corroboration.
                writer_project = memory.project_key or UNSORTED_PROJECT_KEY
                candidate_project = existing.project_key or UNSORTED_PROJECT_KEY
                if writer_project != candidate_project:
                    continue
            if memory.visibility == Visibility.PRIVATE.value and existing.owner_user_id != memory.owner_user_id:
                continue

            # Near-duplicate found, corroborate instead of creating.
            support_relation_outcome = await self._dedup_support_relation_outcome_bundle(
                candidate_memory=memory,
                target_memory=existing,
                doc_id=doc_id,
                source_type=source_type,
                excerpt=excerpt,
                score=score,
            )
            await self.add_source_support(
                existing_id,
                doc_id,
                source_type,
                excerpt,
                support_kind="extracted",
                context=context,
                writer_visibility=memory.visibility,
                writer_owner_user_id=memory.owner_user_id,
                writer_project_key=memory.project_key,
                source_updated_at=source_updated_at,
                relation_outcome=support_relation_outcome,
            )
            logger.debug(
                "Memory corroborated: %s (score=%.4f, doc=%s)",
                existing_id,
                score,
                doc_id,
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
            source_updated_at=source_updated_at,
            relation_outcome=relation_outcome,
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

    async def find_access_compatible_exact_candidate(
        self,
        memory: Memory,
        *,
        excluded_memory_ids: set[str] | frozenset[str] = frozenset(),
    ) -> Memory | None:
        """Return an active exact claim without vector or model dependence."""

        return await self.relational.find_active_exact_claim_candidate(
            memory.content_hash,
            visibility=memory.visibility,
            owner_user_id=memory.owner_user_id,
            repo_identifier=memory.repo_identifier,
            excluded_memory_ids=tuple(sorted(excluded_memory_ids)),
        )

    async def find_access_compatible_equivalence_candidates(
        self,
        memory: Memory,
        *,
        excluded_memory_ids: set[str] | frozenset[str] = frozenset(),
        scope: AccessScope | None = None,
        doc_id: str | None = None,
        entity_ids: Sequence[int] = (),
    ) -> tuple[Memory, ...]:
        """Return bounded access-compatible candidates for semantic proof.

        Strict vector proximity is the primary recall channel. Shared-entity
        candidates from other Source Units are a bounded fallback for equivalent
        claims whose wording is too different for the dedup threshold. The engine
        must still prove exact or semantic equivalence before reusing an identity.
        """

        compatible: list[Memory] = []
        compatible_ids: set[str] = set()
        candidate_access = lifecycle_access_context_hash(
            visibility=memory.visibility,
            owner_user_id=memory.owner_user_id,
            project_key=memory.project_key,
            repo_identifier=memory.repo_identifier,
        )

        def add_candidate(candidate: Memory | None) -> None:
            if (
                candidate is None
                or candidate.id in excluded_memory_ids
                or candidate.id in compatible_ids
                or len(compatible) >= DEDUP_CANDIDATE_LIMIT
                or lifecycle_access_context_hash(
                    visibility=candidate.visibility,
                    owner_user_id=candidate.owner_user_id,
                    project_key=candidate.project_key,
                    repo_identifier=candidate.repo_identifier,
                )
                != candidate_access
            ):
                return
            compatible.append(candidate)
            compatible_ids.add(candidate.id)

        reactivation_candidate = await self.relational.find_rebaseline_reactivation_candidate(
            memory.content_hash,
            visibility=memory.visibility,
            owner_user_id=memory.owner_user_id,
            repo_identifier=memory.repo_identifier,
        )
        add_candidate(reactivation_candidate)

        embedding = await self._embed(_memory_embedding_text(memory))
        vector_hits = await self.vector.query(
            embedding,
            scope or _writer_access_scope(memory),
            None,
            DEDUP_CANDIDATE_LIMIT,
        )
        candidate_ids = [
            existing_id
            for existing_id, score in vector_hits
            if existing_id not in excluded_memory_ids
            and self.vector.within_dedup_threshold(self.dedup_threshold, score)
        ]
        ordinary_by_id = {
            candidate.id: candidate
            for candidate in await self.relational.list_active_ordinary_claim_memories(
                candidate_ids
            )
        }
        vector_candidates: list[Memory] = []
        for existing_id, score in vector_hits:
            if existing_id in excluded_memory_ids:
                continue
            if not self.vector.within_dedup_threshold(self.dedup_threshold, score):
                continue
            existing = ordinary_by_id.get(existing_id)
            if existing is None:
                continue
            if existing.visibility != memory.visibility:
                continue
            if lifecycle_access_context_hash(
                visibility=existing.visibility,
                owner_user_id=existing.owner_user_id,
                project_key=existing.project_key,
                repo_identifier=existing.repo_identifier,
            ) != lifecycle_access_context_hash(
                visibility=memory.visibility,
                owner_user_id=memory.owner_user_id,
                project_key=memory.project_key,
                repo_identifier=memory.repo_identifier,
            ):
                continue
            vector_candidates.append(existing)

        entity_candidates: Sequence[Memory] = ()
        if doc_id and entity_ids:
            entity_candidates = (
                await self.relational.find_active_ordinary_claim_memories_by_entities(
                    entity_ids,
                    visibility=memory.visibility,
                    owner_user_id=memory.owner_user_id,
                    repo_identifier=memory.repo_identifier,
                    project_key=memory.project_key,
                    excluded_memory_ids=tuple(sorted(excluded_memory_ids)),
                    excluded_doc_id=doc_id,
                    limit=DEDUP_CANDIDATE_LIMIT,
                )
            )
        for vector_candidate, entity_candidate in zip_longest(
            vector_candidates,
            entity_candidates,
        ):
            add_candidate(vector_candidate)
            add_candidate(entity_candidate)
        return tuple(compatible)

    async def _dedup_support_relation_outcome_bundle(
        self,
        *,
        candidate_memory: Memory,
        target_memory: Memory,
        doc_id: str,
        source_type: str,
        excerpt: str | None,
        score: float | None,
    ) -> RelationOutcomeBundle:
        document = await self.db.get_document(doc_id)
        source_id = document.source if document is not None and document.source else source_type
        doc_revision_id = None
        if document is not None:
            doc_revision_id = document.content_hash or document.version
        unit = EvidenceUnit(
            id=_dedup_support_evidence_unit_id(
                source_id=source_id,
                doc_id=doc_id,
                doc_revision_id=doc_revision_id,
                candidate_memory_id=candidate_memory.id,
                target_memory_id=target_memory.id,
                content_hash=candidate_memory.content_hash,
            ),
            source_id=source_id,
            doc_id=doc_id,
            doc_revision_id=doc_revision_id,
            source_type=source_type,
            source_anchor=candidate_memory.id,
            source_lineage_id=doc_id,
            project_key=candidate_memory.project_key,
            visibility=candidate_memory.visibility,
            owner_user_id=candidate_memory.owner_user_id,
            repo_identifier=candidate_memory.repo_identifier,
            content=candidate_memory.content,
            excerpt=excerpt,
            evidence_provenance=(
                EvidenceContentProvenance.SOURCE_EXCERPT if excerpt else EvidenceContentProvenance.NO_EXCERPT
            ),
            source_metadata={
                "candidate_memory_id": candidate_memory.id,
                "supported_memory_id": target_memory.id,
                "support_kind": "extracted",
            },
            observed_at=datetime.now(timezone.utc).isoformat(),
        )
        relation_run_id = _dedup_support_relation_run_id(unit, target_memory.id)
        reason = "deduplication candidate within threshold"
        decision = RelationDecision(
            candidate_memory_id=target_memory.id,
            relation_type=RelationType.SUPPORTS,
            authority_case=AuthorityCase.INDEPENDENT_SUPPORT,
            confidence=1.0 if score is None else max(0.0, min(1.0, 1.0 - float(score))),
            reason=reason,
            evidence_excerpt=excerpt,
            matched_bucket=CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
            matched_bucket_complete=True,
            classifier_batch_key=relation_run_id,
        )
        lifecycle = MemoryRelationApplyService().derive_lifecycle(unit, [decision])
        candidate = RelationCandidateRecord(
            relation_run_id=relation_run_id,
            evidence_unit_id=unit.id,
            memory_id=target_memory.id,
            bucket=CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
            bucket_rank=0,
            candidate_rank=0,
            score=score,
            is_mandatory=False,
            bucket_complete=True,
            was_checked=True,
            reason=reason,
        )
        relation_run = RelationRunRecord(
            id=relation_run_id,
            evidence_unit_id=unit.id,
            access_context_hash=unit.access_context_hash,
            candidate_count=1,
            mandatory_candidate_count=0,
            checked_candidate_count=1,
            incomplete_mandatory_buckets=(),
            classifier_version="dedup-support-v1",
            lifecycle_action=lifecycle.action,
            review_case=lifecycle.review_case,
            status="applied" if lifecycle.action is LifecycleAction.ATTACH_SUPPORT else "review",
            result_memory_id=target_memory.id,
            audit={
                "source": "MemoryStore.deduplicate_and_insert",
                "candidate_memory_id": candidate_memory.id,
                "target_memory_id": target_memory.id,
                "dedup_score": score,
            },
        )
        relations: tuple[EvidenceRelationRecord, ...] = ()
        if lifecycle.action is not LifecycleAction.ATTACH_SUPPORT:
            return RelationOutcomeBundle(
                evidence_unit=unit,
                relation_run=relation_run,
                candidates=(candidate,),
                relations=relations,
            )
        relations = (
            EvidenceRelationRecord(
                evidence_unit_id=unit.id,
                memory_id=target_memory.id,
                relation_type=RelationType.SUPPORTS,
                authority_case=AuthorityCase.INDEPENDENT_SUPPORT,
                is_authoritative_support=True,
                source_lineage_id=unit.source_lineage_id,
                confidence=decision.confidence,
                reason=reason,
                excerpt=excerpt,
                classifier_version="dedup-support-v1",
                relation_run_id=relation_run_id,
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
        return RelationOutcomeBundle(
            evidence_unit=unit,
            relation_run=relation_run,
            candidates=(candidate,),
            relations=relations,
        )

    async def insert_memory(
        self,
        memory: Memory,
        doc_id: str,
        source_type: str,
        *,
        source_updated_at: datetime | None,
        entity_ids: list[int] | None = None,
        excerpt: str | None = None,
        relation_outcome: RelationOutcomeBundle | None = None,
        review: MemoryReview | None = None,
        related_review_id: str | None = None,
        related_review_reason: str | None = None,
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
            source_updated_at=source_updated_at,
            relation_outcome=relation_outcome,
            review=review,
            related_review_id=related_review_id,
            related_review_reason=related_review_reason,
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
        *,
        source_updated_at: datetime | None,
        relation_outcome: RelationOutcomeBundle | None = None,
        review: MemoryReview | None = None,
        related_review_id: str | None = None,
        related_review_reason: str | None = None,
    ) -> None:
        """Insert memory into SQLite + FTS5 + ChromaDB + link entities and sources."""
        inserted = False
        chroma_upsert_started = False
        try:
            indexed_text = await self._canonical_memory_embedding_text(memory)
            indexed_embedding = await self._embed(indexed_text)
            await self.db.insert_memory_with_source_and_relation(
                memory,
                doc_id=doc_id,
                source_type=source_type,
                excerpt=excerpt,
                entity_ids=entity_ids,
                relation_outcome=relation_outcome,
                source_updated_at=source_updated_at,
                review=review,
                related_review_id=related_review_id,
                related_review_reason=related_review_reason,
            )
            inserted = True
            await self._emit(
                "fts_upsert_committed",
                "committed",
                context=context,
                memory_id=memory.id,
                doc_id=doc_id,
                payload={"operation": "memory_insert"},
            )

            await self._emit(
                "chroma_upsert_attempted",
                "attempted",
                context=context,
                memory_id=memory.id,
                doc_id=doc_id,
                payload={"operation": "memory_insert"},
            )
            chroma_upsert_started = True
            await self.vector.upsert(
                ids=[memory.id],
                embeddings=[indexed_embedding],
                metadatas=[
                    _memory_metadata(
                        memory,
                        embedding_text_hash=embedding_text_hash(indexed_text),
                        extra={"source_doc_id": doc_id},
                    )
                ],
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
    ) -> None:
        """Update a memory's content across all stores."""
        context = self._operation_context()
        previous = await self.db.get_memory(memory_id)
        previous_vector = await self._memory_vector_snapshot(memory_id)
        memory = None
        try:
            await self.db.update_memory_content(memory_id, new_content, new_confidence)

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
                    metadatas=[
                        _memory_metadata(
                            memory,
                            embedding_text_hash=embedding_text_hash(embedding_text),
                        )
                    ],
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
        *,
        replacement_kind: ReplacementKind,
        source_updated_at: datetime | None,
        entity_ids: list[int] | None = None,
        excerpt: str | None = None,
        replacement_reason: str | None = None,
        relation_outcome: RelationOutcomeBundle | None = None,
        carry_revision_sources: bool | None = None,
    ) -> None:
        """Supersede an old memory with a new one, updating all stores.

        The old memory is marked as superseded in SQLite and removed from
        ChromaDB. The new memory is inserted into all three stores (SQLite,
        FTS5, ChromaDB) and linked to entities and sources.
        """
        context = self._operation_context(doc_id=doc_id)
        old_snapshot = await self.db.get_memory(old_memory_id)
        old_source_snapshots = await self.db.get_memory_sources(old_memory_id)
        old_relation_snapshots = await self.db.get_evidence_relations_by_memory(old_memory_id)
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
            embedding_text = await self._canonical_memory_embedding_text(new_memory)
            embedding = await self._embed(embedding_text)
            await self.db.supersede_memory_with_source_and_relation(
                old_memory_id,
                new_memory,
                replacement_kind=replacement_kind,
                doc_id=doc_id,
                source_type=source_type,
                excerpt=excerpt,
                replacement_reason=replacement_reason,
                carry_revision_sources=(
                    replacement_kind == "revision" if carry_revision_sources is None else carry_revision_sources
                ),
                entity_ids=entity_ids,
                source_updated_at=source_updated_at,
                relation_outcome=relation_outcome,
            )
            await self._remove_from_search_indexes(old_memory_id, label="superseded", context=context)
            await self._emit(
                "fts_upsert_committed",
                "committed",
                context=context,
                memory_id=new_memory.id,
                doc_id=doc_id,
                payload={"operation": "memory_supersede_insert"},
            )

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
                    metadatas=[
                        _memory_metadata(
                            new_memory,
                            embedding_text_hash=embedding_text_hash(embedding_text),
                            extra={"source_doc_id": doc_id},
                        )
                    ],
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
                for source in old_source_snapshots:
                    try:
                        await self.db.restore_memory_source_snapshot(source)
                    except Exception as cleanup_exc:
                        rollback_error = rollback_error or cleanup_exc
                for relation in old_relation_snapshots:
                    try:
                        await self.db.restore_evidence_relation_snapshot(relation)
                    except Exception as cleanup_exc:
                        rollback_error = rollback_error or cleanup_exc
            try:
                await self.db.purge_memory(new_memory.id)
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
            old_memory_id,
            new_memory.id,
            new_memory.memory_type,
        )

    async def insert_agent_claim_memory(
        self,
        memory: Memory,
        projection: SourceProjection,
        lifecycle_plan: LifecyclePlan,
        doc_id: str,
        source_type: str,
        *,
        claim_id: str,
        concept_id: str,
        display_anchor: str,
        claim_text: str,
        memory_type: str,
        confidence: float,
        observed_at: datetime,
        source_updated_at: datetime | None,
        citations: list[str] | None = None,
        concept_projection: dict[str, Any] | None = None,
        concept_markdown_body: str | None = None,
        entity_ids: list[int] | None = None,
        excerpt: str | None = None,
        relation_outcome: RelationOutcomeBundle | None = None,
    ) -> None:
        """Commit Agent Knowledge through the common projected-lifecycle seam."""
        context = self._operation_context(doc_id=doc_id)
        try:
            await self.relational.apply_agent_claim_source_projection_lifecycle(
                projection,
                lifecycle_plan,
                memory_id=memory.id,
                relation_outcome=relation_outcome,
                claim_id=claim_id,
                concept_id=concept_id,
                display_anchor=display_anchor,
                claim_text=claim_text,
                memory_type=memory_type,
                confidence=confidence,
                observed_at=observed_at,
                citations=citations,
                concept_projection=concept_projection,
                concept_markdown_body=concept_markdown_body,
            )
            await self.attempt_lifecycle_vector_delivery(lifecycle_plan.id)
            await self._emit(
                "memory_insert_committed",
                "committed",
                context=context,
                memory_id=memory.id,
                doc_id=doc_id,
                support_kind="extracted",
                reason="agent claim memory inserted",
                payload={"content_hash": memory.content_hash, "memory_type": memory.memory_type},
            )
        except Exception as exc:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=memory.id,
                doc_id=doc_id,
                error=str(exc),
                payload={"operation": "agent_claim_projected_lifecycle"},
            )
            raise

    async def supersede_agent_claim_memory(
        self,
        old_memory_id: str,
        new_memory: Memory,
        projection: SourceProjection,
        lifecycle_plan: LifecyclePlan,
        doc_id: str,
        source_type: str,
        *,
        replacement_kind: ReplacementKind,
        claim_id: str,
        concept_id: str,
        display_anchor: str,
        claim_text: str,
        memory_type: str,
        confidence: float,
        observed_at: datetime,
        source_updated_at: datetime | None,
        citations: list[str] | None = None,
        concept_markdown_body: str | None = None,
        entity_ids: list[int] | None = None,
        excerpt: str | None = None,
        replacement_reason: str | None = None,
        relation_outcome: RelationOutcomeBundle | None = None,
    ) -> None:
        """Commit Agent Knowledge replacement through projected lifecycle."""
        del source_type, replacement_kind, entity_ids, excerpt, source_updated_at
        context = self._operation_context(doc_id=doc_id)
        await self._emit(
            "memory_supersede_attempted",
            "attempted",
            context=context,
            memory_id=old_memory_id,
            candidate_id=new_memory.id,
            reason=replacement_reason,
        )
        try:
            await self.relational.apply_agent_claim_source_projection_lifecycle(
                projection,
                lifecycle_plan,
                memory_id=new_memory.id,
                relation_outcome=relation_outcome,
                claim_id=claim_id,
                concept_id=concept_id,
                display_anchor=display_anchor,
                claim_text=claim_text,
                memory_type=memory_type,
                confidence=confidence,
                observed_at=observed_at,
                citations=citations,
                concept_markdown_body=concept_markdown_body,
            )
            await self.attempt_lifecycle_vector_delivery(lifecycle_plan.id)
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
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=new_memory.id,
                doc_id=doc_id,
                error=str(exc),
                payload={"operation": "agent_claim_projected_lifecycle_supersede"},
            )
            raise

        logger.info(
            "Agent claim memory superseded: %s -> %s (%s)",
            old_memory_id,
            new_memory.id,
            new_memory.memory_type,
        )

    async def retire_agent_claim_memory(
        self,
        *,
        old_memory_id: str,
        projection: SourceProjection,
        plan: LifecyclePlan,
        claim_id: str,
        concept_id: str,
        display_anchor: str,
        claim_text: str,
        memory_type: str,
        confidence: float,
        observed_at: datetime,
        concept_markdown_body: str,
    ) -> None:
        """Atomically retire an Agent claim and update its canonical concept."""

        context = self._operation_context(doc_id=concept_id)
        try:
            await self.relational.apply_agent_claim_source_projection_lifecycle(
                projection,
                plan,
                memory_id=old_memory_id,
                relation_outcome=None,
                claim_id=claim_id,
                concept_id=concept_id,
                display_anchor=display_anchor,
                claim_text=claim_text,
                memory_type=memory_type,
                confidence=confidence,
                observed_at=observed_at,
                concept_markdown_body=concept_markdown_body,
            )
            await self.attempt_lifecycle_vector_delivery(plan.id)
            await self._emit(
                "memory_retire_committed",
                "committed",
                context=context,
                memory_id=old_memory_id,
                reason="managed_agent_claim_retirement",
            )
        except Exception as exc:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=old_memory_id,
                error=str(exc),
                payload={"operation": "agent_claim_projected_lifecycle_retire"},
            )
            raise

    async def ensure_agent_claim_memory_projection(
        self,
        memory: Memory,
        doc_id: str,
        source_type: str,
        *,
        excerpt: str | None = None,
        source_updated_at: datetime | None,
    ) -> None:
        """Converge a committed agent-claim memory with its searchable projections.

        Agent claim replacement IDs are deterministic, so a retry can discover
        that the durable DB lifecycle already committed. The retry must still
        make the committed memory searchable before returning: SQLite source and
        FTS rows are idempotent DB projections, while the vector index is the
        external projection that can legitimately need a second write.
        """
        context = self._operation_context(doc_id=doc_id)
        await self.db.add_memory_source(
            memory.id,
            doc_id,
            source_type,
            excerpt,
            source_updated_at=source_updated_at,
        )
        await self.db.rebuild_memory_fts(
            memory.id,
            search_visible_statuses=set(allowed_search_statuses()),
        )
        if memory.status not in set(allowed_search_statuses()):
            return
        try:
            embedding_text = await self._canonical_memory_embedding_text(memory)
            embedding = await self._embed(embedding_text)
            await self._emit(
                "chroma_upsert_attempted",
                "attempted",
                context=context,
                memory_id=memory.id,
                doc_id=doc_id,
                payload={"operation": "agent_claim_projection_repair"},
            )
            await self.vector.upsert(
                ids=[memory.id],
                embeddings=[embedding],
                metadatas=[
                    _memory_metadata(
                        memory,
                        embedding_text_hash=embedding_text_hash(embedding_text),
                        extra={"source_doc_id": doc_id},
                    )
                ],
            )
            await self._emit(
                "chroma_upsert_committed",
                "committed",
                context=context,
                memory_id=memory.id,
                doc_id=doc_id,
                payload={"operation": "agent_claim_projection_repair"},
            )
        except Exception as exc:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=memory.id,
                doc_id=doc_id,
                error=str(exc),
                payload={"index": "chroma", "operation": "agent_claim_projection_repair"},
            )
            raise

    async def reindex_memory_access(self, memory_id: str) -> None:
        """Converge relational and vector search projections after an access move."""
        memory = await self.db.get_memory(memory_id)
        if memory is None:
            return
        context = self._operation_context()
        await self.db.rebuild_memory_fts(
            memory_id,
            search_visible_statuses=set(allowed_search_statuses()),
        )
        if memory.status not in set(allowed_search_statuses()):
            await self._remove_from_search_indexes(
                memory_id,
                label="source_access_transition",
                context=context,
            )
            return
        try:
            embedding_text = await self._canonical_memory_embedding_text(memory)
            embedding = await self._embed(embedding_text)
            await self.vector.upsert(
                ids=[memory.id],
                embeddings=[embedding],
                metadatas=[
                    _memory_metadata(
                        memory,
                        embedding_text_hash=embedding_text_hash(embedding_text),
                    )
                ],
            )
        except Exception as exc:
            await self._emit(
                "index_operation_failed",
                "failed",
                context=context,
                memory_id=memory.id,
                error=str(exc),
                payload={"index": "chroma", "operation": "source_access_transition"},
            )
            raise

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
        writer_visibility: str | None = None,
        writer_owner_user_id: str | None = None,
        writer_project_key: str | None = None,
        source_updated_at: datetime | None,
        relation_outcome: RelationOutcomeBundle | None = None,
    ) -> str:
        """Add or update source support for an existing memory.

        The mutating boundary for support edges. Outcomes are
        ``"inserted"``, ``"updated"``, ``"unchanged"``, ``"missing"``,
        or ``"rejected"`` when the writer's visibility, owner, or
        project does not match the target. Support edges never cross
        a visibility, owner, or project boundary.
        """
        context = context or self._operation_context(doc_id=doc_id)
        target = await self.db.get_memory(memory_id)
        if target is None:
            return "missing"
        if writer_visibility is not None and writer_visibility != target.visibility:
            return "rejected"
        if writer_visibility == Visibility.PRIVATE.value and writer_owner_user_id != target.owner_user_id:
            return "rejected"
        if writer_visibility == Visibility.WORKSPACE.value:
            # NULL project_key is normalized to UNSORTED at persistence time, so
            # apply the same normalization on both sides of the comparison to
            # keep the workspace-project boundary symmetric across writers and
            # targets.
            expected = writer_project_key or UNSORTED_PROJECT_KEY
            actual = target.project_key or UNSORTED_PROJECT_KEY
            if expected != actual:
                return "rejected"
        await self._emit(
            "source_support_add_attempted",
            "attempted",
            context=context,
            memory_id=memory_id,
            doc_id=doc_id,
            support_kind=support_kind,
        )
        if relation_outcome is not None:
            outcome = await self.db.corroborate_memory_with_relation_outcome(
                memory_id,
                doc_id,
                source_type,
                excerpt,
                support_kind=support_kind,
                source_updated_at=source_updated_at,
                relation_outcome=relation_outcome,
            )
        else:
            outcome = await self.db.corroborate_memory(
                memory_id,
                doc_id,
                source_type,
                excerpt,
                support_kind=support_kind,
                source_updated_at=source_updated_at,
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
        *,
        source_id: str,
        reason: str = "no_support",
        context: AuditContext | None = None,
    ) -> bool:
        """Remove one source link and retire/hide the memory if support reaches zero."""
        context = context or self._operation_context(doc_id=doc_id)
        retired = await self.db.remove_memory_source(
            memory_id,
            doc_id,
            source_id=source_id,
            retire_reason=reason,
        )
        if retired:
            vector_delivery = await self.attempt_lifecycle_vector_delivery(source_id=source_id)
            await self._emit(
                "source_support_removal_retired_memory",
                "committed",
                context=context,
                memory_id=memory_id,
                doc_id=doc_id,
                reason=reason,
                payload={"vector_delivery": vector_delivery.state.value},
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
        memory_ids = await self._memory_ids_for_doc(doc_id)
        memory_snapshots = await self._memory_snapshots(memory_ids)
        source_snapshots = await self._source_snapshots(memory_ids)
        try:
            retired_ids = await self.db.delete_document(doc_id)
        except Exception:
            await self._restore_deleted_document_state(
                document_snapshot=document_snapshot,
                document_side_snapshot=document_side_snapshot,
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

    async def delete_projected_document(
        self,
        doc_id: str,
        *,
        deletion_context: dict[str, Any] | None = None,
    ) -> None:
        """Remove document storage after lifecycle was committed separately.

        This path deliberately performs no Memory mutation. Source Projection
        lineage and Evidence records remain durable for audit while relational
        document artifacts are removed with rollback protection.
        """

        context = self._operation_context(doc_id=doc_id)
        document_snapshot = await self.db.get_document(doc_id)
        document_side_snapshot = await self.db.get_document_side_table_snapshots([doc_id])
        try:
            await self.db.delete_projected_document(doc_id)
        except Exception:
            if document_snapshot:
                await self.db.restore_document_snapshot(
                    document_snapshot,
                    require_configured_source=True,
                )
                await self.db.restore_document_side_table_snapshots(document_side_snapshot)
            raise
        await self._emit(
            "projected_document_delete_committed",
            "committed",
            context=context,
            doc_id=doc_id,
            payload=deletion_context or {},
        )

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
        memory_ids = await self._memory_ids_for_docs(doc_ids)
        memory_snapshots = await self._memory_snapshots(memory_ids)
        source_snapshots = await self._source_snapshots(memory_ids)
        try:
            deletion_result = await self.db.delete_source_cascade(source_id)
        except Exception:
            await self._restore_deleted_source_state(
                source_snapshot=source_snapshot,
                document_snapshots=document_snapshots,
                document_side_snapshot=document_side_snapshot,
                memory_snapshots=memory_snapshots,
                source_snapshots=source_snapshots,
                context=context,
            )
            raise
        retired_ids = list(deletion_result.retired_memory_ids)
        if deletion_result.retired_search_cleanup_required:
            delivery = await self.attempt_lifecycle_vector_delivery(source_id=source_id)
            if delivery.pending:
                # Relational deletion is already complete and authoritative.
                # The independent outbox survives source-row deletion and is
                # retried; never resurrect a partial copy of the deleted graph.
                logger.warning(
                    "Deferred source deletion vector cleanup for %s error_types=%s",
                    source_id,
                    delivery.error_types,
                )
                await self._emit(
                    "source_delete_vector_cleanup_deferred",
                    "failed",
                    context=context,
                    source_id=source_id,
                    error=",".join(delivery.error_types),
                    payload={
                        "retired_memory_ids": retired_ids,
                        "failed_tasks": delivery.failed_tasks,
                    },
                )
        await self._emit(
            "source_delete_cascade_committed",
            "committed",
            context=context,
            source_id=source_id,
            payload={"retired_memory_ids": retired_ids},
        )
        return retired_ids

    async def rebaseline_source_lifecycle(
        self,
        source_id: str,
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> list[str]:
        """Reset replayable source derivations and remove retired vectors."""

        context = self._operation_context(source_id=source_id)
        result = await self.relational.rebaseline_source_lifecycle(
            source_id,
            source_activity=source_activity,
        )
        retired_ids = list(result.retired_memory_ids)
        if result.retired_search_cleanup_required:
            # SQLite records these external vector deletes in the durable
            # lifecycle outbox as part of the relational reset.  Retry earlier
            # failures for this source without coupling its job to another
            # source's pending vector work.
            await self.attempt_lifecycle_vector_delivery(source_id=source_id)
        await self._emit(
            "source_rebaseline_committed",
            "committed",
            context=context,
            source_id=source_id,
            payload={"retired_memory_ids": retired_ids},
        )
        return retired_ids

    async def mark_pending_review(
        self,
        memory_id: str,
        reason: str | None = None,
        *,
        relation_outcome: RelationOutcomeBundle | None = None,
    ) -> None:
        """Quarantine a memory until a human or future workflow resolves it."""
        context = self._operation_context()
        previous = await self.db.get_memory(memory_id)
        search_removed = False
        try:
            await self._remove_from_search_indexes(memory_id, label="pending_review", context=context)
            search_removed = True
            if relation_outcome is not None:
                await self.db.update_memory_status_with_relation_outcome(
                    memory_id,
                    "pending_review",
                    reason=reason,
                    relation_outcome=relation_outcome,
                )
            else:
                await self.db.update_memory_status(memory_id, "pending_review", reason=reason)
        except Exception:
            if previous:
                await self._restore_memory_row(previous)
                if search_removed:
                    await self._restore_search_indexes(previous, context=context, label="pending_review_rollback")
            raise
        await self._emit(
            "memory_pending_review_committed",
            "committed",
            context=context,
            memory_id=memory_id,
            reason=reason,
        )

    async def mark_pending_review_with_case(
        self,
        memory_id: str,
        reason: str | None = None,
        *,
        relation_outcome: RelationOutcomeBundle | None = None,
        review: MemoryReview | None = None,
        related_review_id: str | None = None,
    ) -> None:
        """Quarantine a memory and create/link its review work item as one DB mutation."""
        context = self._operation_context()
        previous = await self.db.get_memory(memory_id)
        await self._remove_from_search_indexes(memory_id, label="pending_review", context=context)
        try:
            await self.db.mark_memory_pending_review_with_case(
                memory_id,
                reason=reason,
                relation_outcome=relation_outcome,
                review=review,
                related_review_id=related_review_id,
            )
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
        replacement_kind: ReplacementKind,
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
                replacement_kind=replacement_kind,
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
                metadatas=[
                    _memory_metadata(
                        challenger,
                        embedding_text_hash=embedding_text_hash(embedding_text),
                        extra={"status": "active"},
                    )
                ],
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
                metadatas=[
                    _memory_metadata(
                        memory,
                        embedding_text_hash=embedding_text_hash(embedding_text),
                    )
                ],
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
        """Return memories linked to a document through the storage contract."""
        return await self.db.get_memory_ids_for_doc(doc_id)

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
        memory_snapshots: list[Memory],
        source_snapshots,
        context: AuditContext,
    ) -> None:
        if document_snapshot:
            await self.db.restore_document_snapshot(
                document_snapshot,
                require_configured_source=(
                    document_snapshot.source not in VIRTUAL_DOCUMENT_SOURCE_IDS
                ),
            )
            await self.db.restore_document_side_table_snapshots(document_side_snapshot)
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
        memory_snapshots: list[Memory],
        source_snapshots,
        context: AuditContext,
    ) -> None:
        if source_snapshot:
            await self.db.restore_source_snapshot(source_snapshot)
        for document in document_snapshots:
            await self.db.restore_document_snapshot(
                document,
                require_configured_source=True,
            )
        await self.db.restore_document_side_table_snapshots(document_side_snapshot)
        for memory in memory_snapshots:
            await self._restore_memory_row(memory)
            await self._restore_search_indexes(memory, context=context, label="source_delete_rollback")
        for source in source_snapshots:
            await self.db.restore_memory_source_snapshot(source)

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
                metadatas=[_normalize_snapshot_metadata(snapshot.get("metadata") or {})],
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
