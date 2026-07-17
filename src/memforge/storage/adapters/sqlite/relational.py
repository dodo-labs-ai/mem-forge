"""SqliteRelationalStore: source-of-truth rows and the scoped read channels.

Row writes and their co-transactional FTS writes stay inside the Database
methods, so this store delegates rather than relocating SQL. The graph,
source/date, visibility, and ranking reads own the SQL that callers run inline
today, so no caller reaches a connection directly.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime
from typing import Any, Mapping, Sequence

import aiosqlite

from memforge.memory.audit import MemoryAuditLogger
from memforge.memory.evidence import (
    ActiveSupportEvidence,
    EvidenceReference,
    MemorySupportAssertion,
    RelationOutcomeBundle,
)
from memforge.memory.lifecycle_plan import (
    ExactRevisionReplay,
    LegacyMemoryProvenance,
    LifecycleCutoverFinding,
    LifecycleBackfillJob,
    CutoverFindingStatus,
    LifecycleGate,
    LifecyclePlan,
    LifecycleReview,
    LifecycleReviewStatus,
    LifecycleVectorTask,
)
from memforge.models import (
    DocumentRecord,
    Entity,
    EntityAlias,
    Memory,
    MemorySource,
    Project,
    SourceLifecycleResetResult,
    Visibility,
    canonicalize_entity_name,
)
from memforge.retrieval.access_predicate import visible_sql
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
from memforge.source_projection import (
    SourceObservationRevision,
    SourceProjection,
    SourceUnit,
    SourceUnitInventoryFilter,
    SourceUnitInventoryPage,
    SourceUnitRevision,
)
from memforge.storage.database import Database
from memforge.storage.adapters.context import AccessScope
from memforge.storage.adapters.protocols import (
    DEFAULT_ENTITY_LINK_LIMIT,
    EntityLinkCandidate,
    EntityLinkResult,
)

logger = logging.getLogger(__name__)

__all__ = ["SqliteRelationalStore"]

# The IN (...) chunk size for the visibility and ranking readers. It carries
# over the value the inline search loops use today so SQLite's bound-parameter
# limit is never exceeded for a large fused candidate set.
_BATCH_SIZE = 200

# The 1-hop graph expansion only keeps memories sharing at least this many
# entities with a direct hit, matching the existing HAVING clause.
_MIN_SHARED_ENTITIES_FOR_EXPANSION = 2

# The 1-hop expansion contributes at half the weight of a direct entity hit.
_EXPANSION_WEIGHT = 0.5

_MAX_ENTITY_LINK_QUERY_TOKENS = 48
_MAX_ENTITY_LINK_WINDOW_TOKENS = 6
_MAX_ENTITY_LINK_WINDOWS = 128
_ENTITY_LINK_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "why",
        "with",
    }
)
_ENTITY_LINK_CHANNEL_SCORE = {
    "explicit": 1.0,
    "alias_exact": 0.95,
    "alias_fts": 0.70,
    "alias_compact": 0.35,
}
_ENTITY_LINK_CHANNEL_ACTIVATES_GRAPH = {
    "explicit": True,
    "alias_exact": True,
    "alias_fts": True,
    "alias_compact": False,
}
_ENTITY_SPECIFICITY_FULL_WEIGHT_MAX_MEMORIES = 8
_ENTITY_SPECIFICITY_BROAD_ENTITY_MIN_MEMORIES = 100
_ENTITY_SPECIFICITY_NORMALIZATION = 0.30
_ENTITY_SPECIFICITY_BROAD_ENTITY_CAP = 0.50


def _entity_specificity(visible_memory_count: int) -> float:
    if visible_memory_count <= 0:
        return 0.0
    if visible_memory_count <= _ENTITY_SPECIFICITY_FULL_WEIGHT_MAX_MEMORIES:
        return 1.0
    raw = 1.0 / math.log2(2 + visible_memory_count)
    normalized = raw / _ENTITY_SPECIFICITY_NORMALIZATION
    # Keep the transition smooth near the full-weight cutoff while preventing
    # very broad team/source entities from contributing like specific terms.
    if visible_memory_count >= _ENTITY_SPECIFICITY_BROAD_ENTITY_MIN_MEMORIES:
        return min(_ENTITY_SPECIFICITY_BROAD_ENTITY_CAP, normalized)
    return min(1.0, normalized)


def _entity_link_tokens(query: str) -> list[str]:
    normalized = canonicalize_entity_name(query)
    return [token for token in normalized.split() if token][:_MAX_ENTITY_LINK_QUERY_TOKENS]


def _entity_link_windows(tokens: Sequence[str]) -> dict[str, str]:
    windows: dict[str, str] = {}
    for start in range(len(tokens)):
        for size in range(1, _MAX_ENTITY_LINK_WINDOW_TOKENS + 1):
            if len(windows) >= _MAX_ENTITY_LINK_WINDOWS:
                return windows
            end = start + size
            if end > len(tokens):
                break
            window_tokens = list(tokens[start:end])
            if size == 1:
                token = window_tokens[0]
                if token in _ENTITY_LINK_STOPWORDS or len(token) < 3:
                    continue
            window = " ".join(window_tokens)
            windows.setdefault(window, window)
    return windows


def _entity_link_compact_terms(tokens: Sequence[str]) -> dict[str, str]:
    terms: dict[str, str] = {}
    for window, matched_text in _entity_link_windows(tokens).items():
        compact = window.replace(" ", "")
        if len(compact) < 4:
            continue
        terms.setdefault(compact, matched_text)
        if compact.endswith("s") and len(compact) > 5:
            terms.setdefault(compact[:-1], matched_text)
    return terms


def _entity_link_fts_query(tokens: Sequence[str]) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in _ENTITY_LINK_STOPWORDS or len(token) < 3:
            continue
        if token in seen:
            continue
        seen.add(token)
        escaped = token.replace('"', '""')
        terms.append(f'"{escaped}"')
    # alias_fts is trusted enough to seed graph retrieval, so require more
    # than a single generic noun. Exact and compact channels still cover
    # single-surface formatting cases separately.
    if len(terms) < 2:
        return ""
    return " OR ".join(terms)


def _entity_link_fts_token_set(tokens: Sequence[str]) -> set[str]:
    return {
        token
        for token in tokens
        if token not in _ENTITY_LINK_STOPWORDS and len(token) >= 3
    }


def _explicit_entity_terms(explicit_entities: Sequence[str]) -> dict[str, str]:
    terms: dict[str, str] = {}
    for value in explicit_entities:
        normalized = canonicalize_entity_name(value)
        if normalized:
            terms.setdefault(normalized, value)
    return terms


def _enabled_source_visibility_condition(
    disabled_source_ids: list[str],
) -> tuple[str | None, list[str]]:
    if not disabled_source_ids:
        return None, []
    placeholders = ", ".join("?" for _ in disabled_source_ids)
    return (
        f"""(
            NOT EXISTS (
                SELECT 1
                FROM memory_sources ms_any
                WHERE ms_any.memory_id = m.id
            )
            OR EXISTS (
                SELECT 1
                FROM memory_sources ms_enabled
                WHERE ms_enabled.memory_id = m.id
                  AND (ms_enabled.source_id IS NULL OR ms_enabled.source_id NOT IN ({placeholders}))
            )
        )""",
        list(disabled_source_ids),
    )


def _append_source_time_predicates(
    *,
    source_filter: MemorySourceFilter,
    time_range: MemoryTimeRange | None,
    joins: list[str],
    clauses: list[str],
    params: list[Any],
) -> tuple[str, bool]:
    """Append canonical source/time predicates and return the deterministic order.

    `source_updated_at` intentionally lives on the same `memory_sources` row as
    exact source facets such as `source_ids`, so a stale Jira provenance row
    cannot match because a different Confluence row was updated recently.
    """

    has_time_filter = time_range is not None and not time_range.is_empty()
    needs_source_join = (
        bool(source_filter.source_ids)
        or bool(source_filter.clients)
        or (has_time_filter and time_range is not None and time_range.date_type == "source_updated_at")
    )
    needs_document_join = bool(source_filter.clients)

    if needs_source_join:
        joins.append("JOIN memory_sources ms ON m.id = ms.memory_id")
    if needs_document_join:
        joins.append("LEFT JOIN documents d ON ms.doc_id = d.doc_id")

    if source_filter.source_ids:
        placeholders = ",".join("?" for _ in source_filter.source_ids)
        clauses.append(f"ms.source_id IN ({placeholders})")
        params.extend(source_filter.source_ids)
    if source_filter.clients:
        placeholders = ",".join("?" for _ in source_filter.clients)
        clauses.append(f"d.client IN ({placeholders})")
        params.extend(source_filter.clients)
    if source_filter.repo_identifiers:
        placeholders = ",".join("?" for _ in source_filter.repo_identifiers)
        clauses.append(f"m.repo_identifier IN ({placeholders})")
        params.extend(source_filter.repo_identifiers)

    if has_time_filter and time_range is not None:
        if time_range.date_type == "source_updated_at":
            if time_range.after is not None:
                clauses.append("ms.source_updated_at >= ?")
                params.append(time_range.after.isoformat())
            if time_range.before is not None:
                clauses.append("ms.source_updated_at < ?")
                params.append(time_range.before.isoformat())
        elif time_range.date_type == "memory_updated_at":
            if time_range.after is not None:
                clauses.append("m.updated_at >= ?")
                params.append(time_range.after.isoformat())
            if time_range.before is not None:
                clauses.append("m.updated_at < ?")
                params.append(time_range.before.isoformat())
        else:
            raise ValueError(f"Unsupported memory time range date_type: {time_range.date_type}")

    if has_time_filter and time_range is not None and time_range.date_type == "source_updated_at":
        return "MAX(ms.source_updated_at) DESC, m.id DESC", needs_source_join
    return "m.updated_at DESC, m.id DESC", needs_source_join


def _source_count_sql(
    *,
    has_source_row_join: bool,
    disabled_source_ids: Sequence[str],
) -> tuple[str, str, list[str]]:
    if has_source_row_join:
        return "", "COUNT(DISTINCT ms.source_id)", []
    if not disabled_source_ids:
        return (
            "LEFT JOIN memory_sources ms_count ON m.id = ms_count.memory_id",
            "COUNT(DISTINCT ms_count.source_id)",
            [],
        )
    placeholders = ", ".join("?" for _ in disabled_source_ids)
    return (
        "LEFT JOIN memory_sources ms_count ON m.id = ms_count.memory_id "
        f"AND (ms_count.source_id IS NULL OR ms_count.source_id NOT IN ({placeholders}))",
        "COUNT(DISTINCT ms_count.source_id)",
        list(disabled_source_ids),
    )


def _entity_link_bound_params(
    *,
    pre_source_count_params: Sequence[Any],
    source_count_params: Sequence[Any],
    post_source_count_params: Sequence[Any],
    limit: int,
) -> list[Any]:
    """Bind entity-link SQL in the same order its placeholders appear.

    `visible_source_count` is projected in the SELECT list, so its placeholders
    must come after CTE parameters but before FROM/WHERE filter parameters.
    """
    return [
        *pre_source_count_params,
        *source_count_params,
        *post_source_count_params,
        limit,
    ]


class SqliteRelationalStore:
    """The row channel backed by the memories table."""

    def __init__(
        self,
        db: Database,
        audit_logger: MemoryAuditLogger | None = None,
    ) -> None:
        self._db = db
        self._audit_logger = audit_logger

    async def insert_memory(self, memory: Memory) -> str:
        return await self._db.insert_memory(memory)

    async def get_memory(self, memory_id: str) -> Memory | None:
        return await self._db.get_memory(memory_id)

    async def find_rebaseline_reactivation_candidate(
        self,
        content_hash: str,
        *,
        visibility: str,
        owner_user_id: str | None,
        repo_identifier: str | None,
    ) -> Memory | None:
        return await self._db.find_rebaseline_reactivation_candidate(
            content_hash,
            visibility=visibility,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
        )

    async def get_exact_revision_replay(
        self,
        *,
        source_id: str,
        source_unit_id: str,
        target_unit_revision_id: str,
        observation_revision_ids: tuple[str, ...],
    ) -> ExactRevisionReplay | None:
        return await self._db.get_exact_revision_replay(
            source_id=source_id,
            source_unit_id=source_unit_id,
            target_unit_revision_id=target_unit_revision_id,
            observation_revision_ids=observation_revision_ids,
        )

    async def get_memory_sources(self, memory_id: str) -> list[MemorySource]:
        return await self._db.get_memory_sources(memory_id)

    async def upsert_document(
        self,
        doc: DocumentRecord,
        *,
        require_configured_source: bool = False,
    ) -> None:
        await self._db.upsert_document(
            doc,
            require_configured_source=require_configured_source,
        )

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        return await self._db.get_document(doc_id)

    async def delete_projected_document(self, doc_id: str) -> None:
        await self._db.delete_projected_document(doc_id)

    async def rebaseline_source_lifecycle(self, source_id: str) -> SourceLifecycleResetResult:
        return await self._db.rebaseline_source_lifecycle(source_id)

    async def rebind_projected_document_support(
        self,
        old_doc_id: str,
        new_doc_id: str,
    ) -> None:
        await self._db.rebind_projected_document_support(old_doc_id, new_doc_id)

    async def record_source_projection(
        self,
        projection: SourceProjection,
        *,
        expected_source_activity_epoch: int | None = None,
    ) -> None:
        await self._db.record_source_projection(
            projection,
            expected_source_activity_epoch=expected_source_activity_epoch,
        )

    async def get_source_projection(self, run_id: str) -> SourceProjection | None:
        return await self._db.get_source_projection(run_id)

    async def create_projection_scope_transition(self, transition):
        return await self._db.create_projection_scope_transition(transition)

    async def get_open_projection_scope_transition(self, source_id: str):
        return await self._db.get_open_projection_scope_transition(source_id)

    async def list_projection_scope_transitions(self, source_id: str, *, limit: int = 20):
        return await self._db.list_projection_scope_transitions(source_id, limit=limit)

    async def start_projection_scope_transition(self, transition_id: str, *, run_id: str):
        return await self._db.start_projection_scope_transition(transition_id, run_id=run_id)

    async def complete_projection_scope_transition(
        self, transition_id: str, *, run_id: str, coverage
    ):
        return await self._db.complete_projection_scope_transition(
            transition_id, run_id=run_id, coverage=coverage
        )

    async def fail_projection_scope_transition(
        self, transition_id: str, *, run_id: str, coverage, error: str
    ):
        return await self._db.fail_projection_scope_transition(
            transition_id, run_id=run_id, coverage=coverage, error=error
        )

    async def get_current_source_unit_revision(
        self,
        source_unit_id: str,
    ) -> SourceUnitRevision | None:
        return await self._db.get_current_source_unit_revision(source_unit_id)

    async def get_current_source_observation_revisions(
        self,
        source_unit_id: str,
    ) -> dict[str, SourceObservationRevision]:
        return dict(await self._db.get_current_source_observation_revisions(source_unit_id))

    async def find_source_unit_by_document_id(
        self,
        source_id: str,
        document_id: str,
        *,
        current_only: bool = False,
    ) -> SourceUnit | None:
        return await self._db.find_source_unit_by_document_id(
            source_id,
            document_id,
            current_only=current_only,
        )

    async def list_source_unit_document_ids(
        self,
        source_unit_id: str,
    ) -> tuple[str, ...]:
        return await self._db.list_source_unit_document_ids(source_unit_id)

    async def list_current_source_unit_observation_ids(
        self,
        source_id: str,
    ) -> dict[str, tuple[str, ...]]:
        return await self._db.list_current_source_unit_observation_ids(source_id)

    async def list_current_source_units(
        self,
        source_id: str,
    ) -> tuple[SourceUnit, ...]:
        return await self._db.list_current_source_units(source_id)

    async def list_current_source_units_page(
        self,
        source_id: str,
        *,
        filters: SourceUnitInventoryFilter,
        cursor: str | None = None,
        limit: int = 200,
    ) -> SourceUnitInventoryPage:
        return await self._db.list_current_source_units_page(
            source_id,
            filters=filters,
            cursor=cursor,
            limit=limit,
        )

    async def list_legacy_memory_provenance(
        self,
        source_id: str,
    ) -> list[LegacyMemoryProvenance]:
        return await self._db.list_legacy_memory_provenance(source_id)

    async def count_active_source_memories_without_support(self, source_id: str) -> int:
        return await self._db.count_active_source_memories_without_support(source_id)

    async def count_active_source_memories(self, source_id: str) -> int:
        return await self._db.count_active_source_memories(source_id)

    async def get_lifecycle_gate(self, source_id: str) -> LifecycleGate:
        return await self._db.get_lifecycle_gate(source_id)

    async def enable_lifecycle_gate(self, source_id: str) -> LifecycleGate:
        return await self._db.enable_lifecycle_gate(source_id)

    async def gate_destructive_lifecycle(self, source_id: str, *, reason: str) -> LifecycleGate:
        return await self._db.gate_destructive_lifecycle(source_id, reason=reason)

    async def upsert_lifecycle_cutover_finding(self, finding: LifecycleCutoverFinding) -> None:
        await self._db.upsert_lifecycle_cutover_finding(finding)

    async def get_lifecycle_cutover_finding(
        self,
        finding_id: str,
    ) -> LifecycleCutoverFinding | None:
        return await self._db.get_lifecycle_cutover_finding(finding_id)

    async def list_lifecycle_cutover_findings(
        self,
        source_id: str,
        *,
        status: CutoverFindingStatus | None = None,
    ) -> list[LifecycleCutoverFinding]:
        return await self._db.list_lifecycle_cutover_findings(source_id, status=status)

    async def create_lifecycle_backfill_job(
        self,
        job: LifecycleBackfillJob,
    ) -> LifecycleBackfillJob:
        return await self._db.create_lifecycle_backfill_job(job)

    async def create_source_rebaseline_job(
        self,
        job: LifecycleBackfillJob,
    ) -> LifecycleBackfillJob:
        return await self._db.create_source_rebaseline_job(job)

    async def start_lifecycle_backfill_job(self, job_id: str) -> LifecycleBackfillJob:
        return await self._db.start_lifecycle_backfill_job(job_id)

    async def complete_lifecycle_backfill_job(
        self,
        job_id: str,
        *,
        scanned_memories: int,
        mapped_memories: int,
        finding_count: int,
    ) -> LifecycleBackfillJob:
        return await self._db.complete_lifecycle_backfill_job(
            job_id,
            scanned_memories=scanned_memories,
            mapped_memories=mapped_memories,
            finding_count=finding_count,
        )

    async def fail_lifecycle_backfill_job(
        self,
        job_id: str,
        *,
        error: str,
    ) -> LifecycleBackfillJob:
        return await self._db.fail_lifecycle_backfill_job(job_id, error=error)

    async def recover_stale_lifecycle_backfill_job(
        self,
        job_id: str,
        *,
        error: str,
    ) -> LifecycleBackfillJob:
        return await self._db.recover_stale_lifecycle_backfill_job(
            job_id,
            error=error,
        )

    async def get_lifecycle_backfill_job(self, job_id: str) -> LifecycleBackfillJob | None:
        return await self._db.get_lifecycle_backfill_job(job_id)

    async def get_active_lifecycle_backfill_job(
        self,
        source_id: str,
    ) -> LifecycleBackfillJob | None:
        return await self._db.get_active_lifecycle_backfill_job(source_id)

    async def list_lifecycle_backfill_jobs(
        self,
        source_id: str,
        *,
        limit: int = 20,
    ) -> list[LifecycleBackfillJob]:
        return await self._db.list_lifecycle_backfill_jobs(source_id, limit=limit)

    async def resolve_lifecycle_cutover_finding(
        self,
        finding_id: str,
        *,
        observation_id: str,
        source_unit_id: str,
    ) -> LifecycleCutoverFinding:
        return await self._db.resolve_lifecycle_cutover_finding(
            finding_id,
            observation_id=observation_id,
            source_unit_id=source_unit_id,
        )

    async def retire_unprovable_lifecycle_cutover_finding(
        self,
        finding_id: str,
        *,
        source_id: str,
        reconstruction_attempt_id: str,
        operator_id: str,
        unavailable_documents: Mapping[str, str],
    ) -> LifecycleCutoverFinding:
        return await self._db.retire_unprovable_lifecycle_cutover_finding(
            finding_id,
            source_id=source_id,
            reconstruction_attempt_id=reconstruction_attempt_id,
            operator_id=operator_id,
            unavailable_documents=unavailable_documents,
        )

    async def record_evidence_references(
        self,
        evidence_unit_id: str,
        references: Sequence[EvidenceReference],
    ) -> tuple[EvidenceReference, ...]:
        return await self._db.record_evidence_references(evidence_unit_id, references)

    async def upsert_memory_support_assertion(self, assertion: MemorySupportAssertion) -> None:
        await self._db.upsert_memory_support_assertion(assertion)

    async def get_memory_support_set_hash(self, memory_id: str) -> str:
        return await self._db.get_memory_support_set_hash(memory_id)

    async def get_active_memory_support_reference_ids(self, memory_id: str) -> tuple[str, ...]:
        return await self._db.get_active_memory_support_reference_ids(memory_id)

    async def get_active_memory_support_evidence(
        self,
        memory_id: str,
        *,
        source_id: str | None = None,
    ) -> tuple[ActiveSupportEvidence, ...]:
        return await self._db.get_active_memory_support_evidence(
            memory_id,
            source_id=source_id,
        )

    async def get_source_unit_support_reference_ids(
        self,
        source_unit_id: str,
    ) -> dict[str, tuple[str, ...]]:
        return dict(await self._db.get_source_unit_support_reference_ids(source_unit_id))

    async def apply_source_projection_lifecycle(
        self,
        projection: SourceProjection,
        plan: LifecyclePlan,
        *,
        expected_source_activity_epoch: int | None = None,
    ) -> None:
        await self._db.apply_source_projection_lifecycle(
            projection,
            plan,
            expected_source_activity_epoch=expected_source_activity_epoch,
        )

    async def apply_agent_claim_source_projection_lifecycle(
        self,
        projection: SourceProjection,
        plan: LifecyclePlan,
        *,
        memory_id: str,
        relation_outcome: RelationOutcomeBundle | None,
        claim_id: str,
        concept_id: str,
        display_anchor: str,
        claim_text: str,
        memory_type: str,
        tags: list[str],
        confidence: float,
        observed_at: datetime,
        citations: list[str] | None = None,
        concept_projection: Mapping[str, object] | None = None,
        concept_markdown_body: str | None = None,
    ) -> None:
        await self._db.apply_agent_claim_source_projection_lifecycle(
            projection,
            plan,
            memory_id=memory_id,
            relation_outcome=relation_outcome,
            claim_id=claim_id,
            concept_id=concept_id,
            display_anchor=display_anchor,
            claim_text=claim_text,
            memory_type=memory_type,
            tags=tags,
            confidence=confidence,
            observed_at=observed_at,
            citations=citations,
            concept_projection=dict(concept_projection) if concept_projection is not None else None,
            concept_markdown_body=concept_markdown_body,
        )

    async def apply_lifecycle_plan(self, plan: LifecyclePlan) -> None:
        await self._db.apply_lifecycle_plan(plan)

    async def get_lifecycle_plan_payload(
        self,
        lifecycle_plan_id: str,
    ) -> Mapping[str, object] | None:
        return await self._db.get_lifecycle_plan_payload(lifecycle_plan_id)

    async def get_lifecycle_review(self, review_id: str) -> LifecycleReview | None:
        return await self._db.get_lifecycle_review(review_id)

    async def list_lifecycle_reviews(
        self,
        source_id: str,
        *,
        status: LifecycleReviewStatus | None = None,
    ) -> list[LifecycleReview]:
        return await self._db.list_lifecycle_reviews(source_id, status=status)

    async def resolve_lifecycle_review(
        self,
        review_id: str,
        status: LifecycleReviewStatus,
    ) -> LifecycleReview:
        return await self._db.resolve_lifecycle_review(review_id, status)

    async def list_lifecycle_vector_tasks(
        self,
        *,
        source_id: str | None = None,
        lifecycle_plan_id: str | None = None,
        limit: int = 100,
    ) -> list[LifecycleVectorTask]:
        return await self._db.list_lifecycle_vector_tasks(
            source_id=source_id,
            lifecycle_plan_id=lifecycle_plan_id,
            limit=limit,
        )

    async def complete_lifecycle_vector_task(self, task_id: str) -> None:
        await self._db.complete_lifecycle_vector_task(task_id)

    async def fail_lifecycle_vector_task(self, task_id: str, error: str) -> None:
        await self._db.fail_lifecycle_vector_task(task_id, error)

    async def get_aliases_for_entity(self, entity_id: int) -> list[EntityAlias]:
        return await self._db.get_aliases_for_entity(entity_id)

    async def get_all_entities(self) -> list[Entity]:
        return await self._db.get_all_entities()

    async def get_all_aliases(self) -> list[tuple[str, int]]:
        return await self._db.get_all_aliases()

    async def add_memory_source(
        self,
        memory_id: str,
        doc_id: str,
        source_type: str,
        excerpt: str | None,
        *,
        support_kind: str = "extracted",
        source_updated_at: datetime | None,
    ) -> None:
        await self._db.add_memory_source(
            memory_id,
            doc_id,
            source_type,
            excerpt,
            support_kind=support_kind,
            source_updated_at=source_updated_at,
        )

    async def promote_to_workspace(
        self,
        memory_id: str,
        *,
        actor_user_id: str,
        reason: str,
    ) -> None:
        """Flip a private memory to workspace visibility.

        The full flip-and-redo flow (re-stamping vector metadata in place and
        re-running dedup against the team set) is designed but not yet
        implemented. This method locks the contract: it verifies the row
        exists and is private, that the actor owns it, audits the attempt,
        and then refuses with NotImplementedError. A non-owner caller is
        rejected before any audit row is written, so a hostile attempt
        leaves no trail.
        """
        target = await self.get_memory(memory_id)
        if target is None:
            raise LookupError(f"memory {memory_id!r} not found")
        if target.visibility != Visibility.PRIVATE.value:
            raise ValueError(f"memory {memory_id!r} is not private; nothing to promote")
        if target.owner_user_id != actor_user_id:
            raise PermissionError(f"actor {actor_user_id!r} does not own memory {memory_id!r}")
        if self._audit_logger is not None:
            await self._audit_logger.emit(
                "memory_promoted",
                "failed",
                memory_id=memory_id,
                reason="not_implemented",
                payload={
                    "requested_reason": reason,
                    "actor": actor_user_id,
                },
            )
        raise NotImplementedError("promote_to_workspace is not yet implemented")

    async def filter_visible_ids(self, ids: Sequence[str], scope: AccessScope) -> set[str]:
        visible: set[str] = set()
        memory_ids = list(ids)
        if not memory_ids:
            return visible
        pred_sql, pred_params = visible_sql(scope, "m")
        for start in range(0, len(memory_ids), _BATCH_SIZE):
            batch = memory_ids[start : start + _BATCH_SIZE]
            placeholders = ",".join("?" for _ in batch)
            sql = f"SELECT m.id FROM memories m WHERE m.id IN ({placeholders}) AND {pred_sql}"
            try:
                async with self._db.db.execute(sql, [*batch, *pred_params]) as cursor:
                    async for row in cursor:
                        visible.add(row["id"])
            except Exception:
                logger.exception("Failed to filter visible memory ids")
                return set()
        return visible

    async def filter_ids_by_source_and_time(
        self,
        ids: Sequence[str],
        source_filter: MemorySourceFilter | None = None,
        time_range: MemoryTimeRange | None = None,
    ) -> set[str]:
        memory_ids = list(ids)
        if not memory_ids:
            return set()
        source_filter = source_filter or MemorySourceFilter()
        has_source_filter = not source_filter.is_empty()
        has_time_filter = time_range is not None and not time_range.is_empty()
        if not has_source_filter and not has_time_filter:
            return set(memory_ids)

        matched: set[str] = set()
        for start in range(0, len(memory_ids), _BATCH_SIZE):
            batch = memory_ids[start : start + _BATCH_SIZE]
            id_placeholders = ",".join("?" for _ in batch)
            joins: list[str] = []
            clauses = [f"m.id IN ({id_placeholders})"]
            params: list[Any] = [*batch]

            _append_source_time_predicates(
                source_filter=source_filter,
                time_range=time_range,
                joins=joins,
                clauses=clauses,
                params=params,
            )

            sql = (
                "SELECT DISTINCT m.id "
                "FROM memories m "
                + (" ".join(joins) + " " if joins else "")
                + "WHERE "
                + " AND ".join(clauses)
            )
            try:
                async with self._db.db.execute(sql, params) as cursor:
                    async for row in cursor:
                        matched.add(row[0])
            except Exception:
                logger.exception("Failed to filter ids by structured source/date facets")
                return set()
        return matched

    async def list_ids_by_source_and_time(
        self,
        source_filter: MemorySourceFilter | None,
        time_range: MemoryTimeRange | None,
        scope: AccessScope,
        *,
        limit: int,
        offset: int,
    ) -> tuple[list[str], int]:
        source_filter = source_filter or MemorySourceFilter()
        has_source_filter = not source_filter.is_empty()
        has_time_filter = time_range is not None and not time_range.is_empty()
        if not has_source_filter and not has_time_filter:
            raise ValueError("list_ids_by_source_and_time requires source_filter or time_range")

        predicate_sql, predicate_params = visible_sql(scope, "m")
        joins: list[str] = []
        clauses = [predicate_sql]
        params: list[Any] = list(predicate_params)
        disabled_source_ids = await self._db.list_disabled_source_ids_for_user(scope.user_id)
        order_sql, has_source_row_join = _append_source_time_predicates(
            source_filter=source_filter,
            time_range=time_range,
            joins=joins,
            clauses=clauses,
            params=params,
        )
        if has_source_row_join and disabled_source_ids:
            placeholders = ", ".join("?" for _ in disabled_source_ids)
            clauses.append(f"(ms.source_id IS NULL OR ms.source_id NOT IN ({placeholders}))")
            params.extend(disabled_source_ids)
        else:
            source_visibility_sql, source_visibility_params = _enabled_source_visibility_condition(disabled_source_ids)
            if source_visibility_sql:
                clauses.append(source_visibility_sql)
                params.extend(source_visibility_params)
        join_sql = " ".join(joins)
        where_sql = " AND ".join(clauses)
        group_sql = "GROUP BY m.id" if joins else ""

        count_sql = f"SELECT COUNT(*) FROM (SELECT m.id FROM memories m {join_sql} WHERE {where_sql} {group_sql}) q"
        async with self._db.db.execute(count_sql, params) as cursor:
            row = await cursor.fetchone()
            total = int(row[0]) if row else 0

        page_sql = (
            f"SELECT m.id FROM memories m {join_sql} WHERE {where_sql} "
            f"{group_sql} ORDER BY {order_sql} LIMIT ? OFFSET ?"
        )
        ids: list[str] = []
        async with self._db.db.execute(page_sql, [*params, limit, offset]) as cursor:
            async for row in cursor:
                ids.append(row[0])
        return ids, total

    async def graph_search(
        self,
        entity_ids: Sequence[int],
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
        *,
        source_filter: MemorySourceFilter | None = None,
        time_range: MemoryTimeRange | None = None,
    ) -> list[tuple[str, float]]:
        ids = list(entity_ids)
        if not ids:
            return []

        placeholders = ",".join("?" for _ in ids)
        predicate_sql, predicate_params = visible_sql(scope, "m")
        joins: list[str] = []
        clauses = [predicate_sql]
        params: list[Any] = [*predicate_params]

        source_filter = source_filter or MemorySourceFilter()
        _, has_source_row_join = _append_source_time_predicates(
            source_filter=source_filter,
            time_range=time_range,
            joins=joins,
            clauses=clauses,
            params=params,
        )
        if memory_types:
            type_placeholders = ",".join("?" for _ in memory_types)
            clauses.append(f"m.memory_type IN ({type_placeholders})")
            params.extend(memory_types)

        disabled_source_ids = await self._db.list_disabled_source_ids_for_user(scope.user_id)
        if has_source_row_join and disabled_source_ids:
            disabled_placeholders = ", ".join("?" for _ in disabled_source_ids)
            clauses.append(f"(ms.source_id IS NULL OR ms.source_id NOT IN ({disabled_placeholders}))")
            params.extend(disabled_source_ids)
        else:
            source_visibility_sql, source_visibility_params = _enabled_source_visibility_condition(disabled_source_ids)
            if source_visibility_sql:
                clauses.append(source_visibility_sql)
                params.extend(source_visibility_params)

        join_sql = " ".join(joins)
        where_sql = " AND ".join(clauses)

        direct_sql = (
            "SELECT m.id, COUNT(DISTINCT me.entity_id) AS entity_overlap "
            "FROM memories m "
            "JOIN memory_entities me ON m.id = me.memory_id "
            f"{join_sql} "
            f"WHERE me.entity_id IN ({placeholders}) "
            f"AND {where_sql} "
            "GROUP BY m.id "
            "ORDER BY entity_overlap DESC "
            "LIMIT ?"
        )
        direct_params: list[Any] = [*ids, *params, limit]

        direct_results: list[tuple[str, int]] = []
        try:
            async with self._db.db.execute(direct_sql, direct_params) as cursor:
                async for row in cursor:
                    direct_results.append((row[0], int(row[1])))
        except Exception:
            logger.exception("Graph direct search failed")
            return []

        query_entity_count = len(ids)
        scored: list[tuple[str, float]] = [
            (mid, float(overlap) / query_entity_count) for mid, overlap in direct_results
        ]

        if direct_results:
            direct_ids = [mid for mid, _ in direct_results]
            d_placeholders = ",".join("?" for _ in direct_ids)
            expanded_sql = (
                "SELECT m.id, COUNT(DISTINCT me2.entity_id) AS shared_entities "
                "FROM memory_entities me1 "
                "JOIN memory_entities me2 ON me1.entity_id = me2.entity_id "
                "JOIN memories m ON me2.memory_id = m.id "
                f"{join_sql} "
                f"WHERE me1.memory_id IN ({d_placeholders}) "
                f"AND m.id NOT IN ({d_placeholders}) "
                f"AND {where_sql} "
                "GROUP BY m.id "
                f"HAVING shared_entities >= {_MIN_SHARED_ENTITIES_FOR_EXPANSION} "
                "ORDER BY shared_entities DESC "
                "LIMIT ?"
            )
            expanded_params: list[Any] = [
                *direct_ids,
                *direct_ids,
                *params,
                limit,
            ]
            try:
                async with self._db.db.execute(expanded_sql, expanded_params) as cursor:
                    async for row in cursor:
                        shared = int(row[1])
                        scored.append((row[0], _EXPANSION_WEIGHT * shared / query_entity_count))
            except Exception:
                logger.exception("Graph 1-hop expansion failed")

        return scored

    async def link_query_entities(
        self,
        query: str,
        *,
        scope: AccessScope,
        explicit_entities: Sequence[str] = (),
        source_filter: MemorySourceFilter | None = None,
        time_range: MemoryTimeRange | None = None,
        memory_types: Sequence[str] | None = None,
        limit: int = DEFAULT_ENTITY_LINK_LIMIT,
    ) -> EntityLinkResult:
        max_candidates = max(0, int(limit))
        explicit_terms = _explicit_entity_terms(explicit_entities)
        if max_candidates == 0:
            return EntityLinkResult(unmatched_explicit_entities=tuple(explicit_terms.values()))

        candidates: dict[int, EntityLinkCandidate] = {}
        matched_explicit_terms: set[str] = set()

        async def add_matches(channel: str, terms: dict[str, str]) -> None:
            if not terms:
                return
            rows = await self._lookup_entity_link_rows(
                terms,
                channel=channel,
                scope=scope,
                source_filter=source_filter,
                time_range=time_range,
                memory_types=memory_types,
                limit=max(max_candidates * 4, max_candidates),
            )
            for row in rows:
                entity_id = int(row["entity_id"])
                visible_memory_count = int(row["visible_memory_count"] or 0)
                visible_source_count = int(row["visible_source_count"] or 0)
                if channel == "explicit":
                    matched_explicit_terms.add(str(row["match_key"]))
                score = _ENTITY_LINK_CHANNEL_SCORE[channel]
                activates_graph = (
                    _ENTITY_LINK_CHANNEL_ACTIVATES_GRAPH[channel]
                    and visible_memory_count > 0
                )
                candidate = EntityLinkCandidate(
                    entity_id=entity_id,
                    canonical_name=str(row["canonical_name"]),
                    matched_alias=str(row["matched_alias"]),
                    channel=channel,
                    contributing_channels=(channel,),
                    score=score,
                    matched_text=terms.get(str(row["match_key"]), str(row["match_key"])),
                    activates_graph=activates_graph,
                    visible_memory_count=visible_memory_count,
                    visible_source_count=visible_source_count,
                    specificity=_entity_specificity(visible_memory_count),
                )
                existing = candidates.get(entity_id)
                if existing is None:
                    candidates[entity_id] = candidate
                    continue

                contributing_channels = tuple(
                    dict.fromkeys((*existing.contributing_channels, channel))
                )
                if score > existing.score:
                    candidates[entity_id] = EntityLinkCandidate(
                        entity_id=existing.entity_id,
                        canonical_name=candidate.canonical_name,
                        matched_alias=candidate.matched_alias,
                        channel=channel,
                        contributing_channels=contributing_channels,
                        score=score,
                        matched_text=candidate.matched_text,
                        activates_graph=activates_graph,
                        visible_memory_count=candidate.visible_memory_count,
                        visible_source_count=candidate.visible_source_count,
                        specificity=candidate.specificity,
                    )
                else:
                    candidates[entity_id] = EntityLinkCandidate(
                        entity_id=existing.entity_id,
                        canonical_name=existing.canonical_name,
                        matched_alias=existing.matched_alias,
                        channel=existing.channel,
                        contributing_channels=contributing_channels,
                        score=existing.score,
                        matched_text=existing.matched_text,
                        activates_graph=existing.activates_graph,
                        visible_memory_count=existing.visible_memory_count,
                        visible_source_count=existing.visible_source_count,
                        specificity=existing.specificity,
                    )

        await add_matches("explicit", explicit_terms)

        tokens = _entity_link_tokens(query)
        await add_matches("alias_exact", _entity_link_windows(tokens))
        fts_query = _entity_link_fts_query(tokens)
        if fts_query:
            for row in await self._lookup_entity_fts_rows(
                fts_query,
                query_tokens=_entity_link_fts_token_set(tokens),
                matched_text=" ".join(tokens),
                scope=scope,
                source_filter=source_filter,
                time_range=time_range,
                memory_types=memory_types,
                limit=max(max_candidates * 4, max_candidates),
            ):
                entity_id = int(row["entity_id"])
                channel = "alias_fts"
                visible_memory_count = int(row["visible_memory_count"] or 0)
                visible_source_count = int(row["visible_source_count"] or 0)
                candidate = EntityLinkCandidate(
                    entity_id=entity_id,
                    canonical_name=str(row["canonical_name"]),
                    matched_alias=str(row["matched_alias"]),
                    channel=channel,
                    contributing_channels=(channel,),
                    score=_ENTITY_LINK_CHANNEL_SCORE[channel],
                    matched_text=str(row["matched_text"]),
                    activates_graph=(
                        _ENTITY_LINK_CHANNEL_ACTIVATES_GRAPH[channel]
                        and visible_memory_count > 0
                    ),
                    visible_memory_count=visible_memory_count,
                    visible_source_count=visible_source_count,
                    specificity=_entity_specificity(visible_memory_count),
                )
                existing = candidates.get(entity_id)
                if existing is None:
                    candidates[entity_id] = candidate
                else:
                    contributing_channels = tuple(
                        dict.fromkeys((*existing.contributing_channels, channel))
                    )
                    if candidate.score > existing.score:
                        candidates[entity_id] = EntityLinkCandidate(
                            entity_id=existing.entity_id,
                            canonical_name=candidate.canonical_name,
                            matched_alias=candidate.matched_alias,
                            channel=channel,
                            contributing_channels=contributing_channels,
                            score=candidate.score,
                            matched_text=candidate.matched_text,
                            activates_graph=candidate.activates_graph,
                            visible_memory_count=candidate.visible_memory_count,
                            visible_source_count=candidate.visible_source_count,
                            specificity=candidate.specificity,
                        )
                    else:
                        candidates[entity_id] = EntityLinkCandidate(
                            entity_id=existing.entity_id,
                            canonical_name=existing.canonical_name,
                            matched_alias=existing.matched_alias,
                            channel=existing.channel,
                            contributing_channels=contributing_channels,
                            score=existing.score,
                            matched_text=existing.matched_text,
                            activates_graph=existing.activates_graph,
                            visible_memory_count=existing.visible_memory_count,
                            visible_source_count=existing.visible_source_count,
                            specificity=existing.specificity,
                        )
        await add_matches("alias_compact", _entity_link_compact_terms(tokens))

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: (
                -candidate.score,
                candidate.canonical_name,
                candidate.entity_id,
            ),
        )[:max_candidates]
        unmatched_explicit_entities = tuple(
            raw_value
            for normalized, raw_value in explicit_terms.items()
            if normalized not in matched_explicit_terms
        )
        return EntityLinkResult(
            candidates=tuple(ranked),
            unmatched_explicit_entities=unmatched_explicit_entities,
        )

    async def _lookup_entity_link_rows(
        self,
        terms: dict[str, str],
        *,
        channel: str,
        scope: AccessScope,
        source_filter: MemorySourceFilter | None,
        time_range: MemoryTimeRange | None,
        memory_types: Sequence[str] | None,
        limit: int,
    ) -> list[Any]:
        term_values = list(terms)
        if not term_values:
            return []

        placeholders = ",".join("?" for _ in term_values)
        if channel == "alias_compact":
            entity_match = f"REPLACE(e.canonical_name, ' ', '') IN ({placeholders})"
            alias_match = f"REPLACE(ea.alias_normalized, ' ', '') IN ({placeholders})"
            entity_match_key = "REPLACE(e.canonical_name, ' ', '')"
            alias_match_key = "REPLACE(ea.alias_normalized, ' ', '')"
        else:
            entity_match = f"e.canonical_name IN ({placeholders})"
            alias_match = f"ea.alias_normalized IN ({placeholders})"
            entity_match_key = "e.canonical_name"
            alias_match_key = "ea.alias_normalized"

        predicate_sql, predicate_params = visible_sql(scope, "m")
        joins: list[str] = []
        clauses = [predicate_sql]
        filter_params: list[Any] = [*predicate_params]

        source_filter = source_filter or MemorySourceFilter()
        _, has_source_row_join = _append_source_time_predicates(
            source_filter=source_filter,
            time_range=time_range,
            joins=joins,
            clauses=clauses,
            params=filter_params,
        )

        if memory_types:
            type_placeholders = ",".join("?" for _ in memory_types)
            clauses.append(f"m.memory_type IN ({type_placeholders})")
            filter_params.extend(memory_types)

        disabled_source_ids = await self._db.list_disabled_source_ids_for_user(scope.user_id)
        if has_source_row_join and disabled_source_ids:
            disabled_placeholders = ", ".join("?" for _ in disabled_source_ids)
            clauses.append(f"(ms.source_id IS NULL OR ms.source_id NOT IN ({disabled_placeholders}))")
            filter_params.extend(disabled_source_ids)
        else:
            source_visibility_sql, source_visibility_params = _enabled_source_visibility_condition(disabled_source_ids)
            if source_visibility_sql:
                clauses.append(source_visibility_sql)
                filter_params.extend(source_visibility_params)

        join_sql = " ".join(joins)
        source_count_join, source_count_expr, source_count_params = _source_count_sql(
            has_source_row_join=has_source_row_join,
            disabled_source_ids=disabled_source_ids,
        )
        count_select = (
            "COUNT(DISTINCT m.id) AS visible_memory_count, "
            f"{source_count_expr} AS visible_source_count"
        )
        where_sql = " AND ".join(clauses)
        sql = (
            "WITH matched_aliases(entity_id, matched_alias, alias_normalized, match_key) AS ("
            f"SELECT e.id, e.canonical_name, e.canonical_name, {entity_match_key} "
            f"FROM entities e WHERE {entity_match} "
            "UNION ALL "
            f"SELECT ea.canonical_id, ea.alias, ea.alias_normalized, {alias_match_key} "
            f"FROM entity_aliases ea WHERE {alias_match}"
            ") "
            "SELECT ma.entity_id, e.canonical_name, ma.matched_alias, "
            f"ma.alias_normalized, ma.match_key, {count_select} "
            "FROM matched_aliases ma "
            "JOIN entities e ON e.id = ma.entity_id "
            "JOIN memory_entities me ON me.entity_id = ma.entity_id "
            "JOIN memories m ON m.id = me.memory_id "
            f"{join_sql} "
            f"{source_count_join} "
            f"WHERE {where_sql} "
            "GROUP BY ma.entity_id, e.canonical_name, ma.matched_alias, ma.alias_normalized, ma.match_key "
            "ORDER BY visible_memory_count DESC, LENGTH(ma.alias_normalized) DESC, e.canonical_name ASC "
            "LIMIT ?"
        )
        bound_params = _entity_link_bound_params(
            pre_source_count_params=(*term_values, *term_values),
            source_count_params=source_count_params,
            post_source_count_params=filter_params,
            limit=limit,
        )
        try:
            async with self._db.db.execute(sql, bound_params) as cursor:
                return [row async for row in cursor]
        except (aiosqlite.Error, sqlite3.Error):
            logger.exception("SQLite entity linker query failed")
            return []

    async def _lookup_entity_fts_rows(
        self,
        fts_query: str,
        *,
        query_tokens: set[str],
        matched_text: str,
        scope: AccessScope,
        source_filter: MemorySourceFilter | None,
        time_range: MemoryTimeRange | None,
        memory_types: Sequence[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if len(query_tokens) < 2:
            return []
        predicate_sql, predicate_params = visible_sql(scope, "m")
        joins: list[str] = []
        clauses = [predicate_sql]
        filter_params: list[Any] = [*predicate_params]

        source_filter = source_filter or MemorySourceFilter()
        _, has_source_row_join = _append_source_time_predicates(
            source_filter=source_filter,
            time_range=time_range,
            joins=joins,
            clauses=clauses,
            params=filter_params,
        )

        if memory_types:
            type_placeholders = ",".join("?" for _ in memory_types)
            clauses.append(f"m.memory_type IN ({type_placeholders})")
            filter_params.extend(memory_types)

        disabled_source_ids = await self._db.list_disabled_source_ids_for_user(scope.user_id)
        if has_source_row_join and disabled_source_ids:
            disabled_placeholders = ", ".join("?" for _ in disabled_source_ids)
            clauses.append(f"(ms.source_id IS NULL OR ms.source_id NOT IN ({disabled_placeholders}))")
            filter_params.extend(disabled_source_ids)
        else:
            source_visibility_sql, source_visibility_params = _enabled_source_visibility_condition(disabled_source_ids)
            if source_visibility_sql:
                clauses.append(source_visibility_sql)
                filter_params.extend(source_visibility_params)

        join_sql = " ".join(joins)
        source_count_join, source_count_expr, source_count_params = _source_count_sql(
            has_source_row_join=has_source_row_join,
            disabled_source_ids=disabled_source_ids,
        )
        count_select = (
            "COUNT(DISTINCT m.id) AS visible_memory_count, "
            f"{source_count_expr} AS visible_source_count"
        )
        where_sql = " AND ".join(clauses)
        sql = (
            "WITH matched_aliases AS ("
            "SELECT entity_id, canonical_name, alias_normalized, search_text, rank AS fts_rank "
            "FROM entity_alias_search_fts "
            "WHERE entity_alias_search_fts MATCH ?"
            ") "
            "SELECT ma.entity_id, e.canonical_name, ma.alias_normalized AS matched_alias, "
            f"ma.search_text, MIN(ma.fts_rank) AS best_rank, {count_select} "
            "FROM matched_aliases ma "
            "JOIN entities e ON e.id = ma.entity_id "
            "JOIN memory_entities me ON me.entity_id = ma.entity_id "
            "JOIN memories m ON m.id = me.memory_id "
            f"{join_sql} "
            f"{source_count_join} "
            f"WHERE {where_sql} "
            "GROUP BY ma.entity_id, e.canonical_name, ma.alias_normalized, ma.search_text "
            "ORDER BY best_rank ASC, visible_memory_count DESC, e.canonical_name ASC "
            "LIMIT ?"
        )
        bound_params = _entity_link_bound_params(
            pre_source_count_params=(fts_query,),
            source_count_params=source_count_params,
            post_source_count_params=filter_params,
            limit=limit,
        )
        try:
            async with self._db.db.execute(sql, bound_params) as cursor:
                rows = [dict(row) async for row in cursor]
        except (aiosqlite.Error, sqlite3.Error):
            logger.exception("SQLite entity alias FTS query failed")
            return []
        kept: list[dict[str, Any]] = []
        for row in rows:
            row_tokens = set(_entity_link_tokens(str(row.get("search_text") or "")))
            # FTS rows are one canonical or alias surface. Require the query to
            # overlap that same surface by at least two tokens before treating
            # the match as a graph-activating entity link.
            if len(query_tokens.intersection(row_tokens)) < 2:
                continue
            row["matched_text"] = matched_text
            kept.append(row)
        return kept

    async def fetch_ranking_metadata(self, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Return recency and affinity metadata for each id in one read."""
        ranked: dict[str, dict[str, Any]] = {}
        memory_ids = list(ids)
        for start in range(0, len(memory_ids), _BATCH_SIZE):
            batch = memory_ids[start : start + _BATCH_SIZE]
            placeholders = ",".join("?" for _ in batch)
            try:
                async with self._db.db.execute(
                    "SELECT m.id, m.updated_at, m.project_key, "
                    "m.repo_identifier "
                    "FROM memories m "
                    f"WHERE m.id IN ({placeholders}) "
                    "GROUP BY m.id",
                    batch,
                ) as cursor:
                    async for row in cursor:
                        raw_updated = row[1]
                        parsed: datetime | None = None
                        if raw_updated:
                            try:
                                parsed = datetime.fromisoformat(raw_updated)
                            except (ValueError, TypeError):
                                parsed = None
                        ranked[row[0]] = {
                            "updated_at": parsed,
                            "project_key": row[2],
                            "repo_identifier": row[3],
                        }
            except Exception:
                logger.exception("Failed to fetch ranking metadata for memory ids")
        return ranked

    async def create_project(self, *, key: str, name: str, is_shared: bool = False) -> Project:
        return await self._db.create_project(key=key, name=name, is_shared=is_shared)

    async def get_project(self, project_id: str) -> Project | None:
        return await self._db.get_project(project_id)

    async def list_projects(self) -> list[Project]:
        return await self._db.list_projects()

    async def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        is_shared: bool | None = None,
    ) -> Project | None:
        return await self._db.update_project(project_id, name=name, is_shared=is_shared)

    async def list_project_memory_ids(self, project_id: str) -> list[str]:
        return await self._db.list_project_memory_ids(project_id)

    async def commit_project_deletion(self, project_id: str, affected_ids: Sequence[str]) -> None:
        await self._db.commit_project_deletion(project_id, affected_ids)
