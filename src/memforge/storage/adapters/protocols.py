"""Storage adapter protocols: the three narrow contracts the core binds to.

RelationalStore is the source-of-truth rows, KeywordSearch is the BM25/FTS5
channel, VectorStore is the embedding channel. Enforcement is each
adapter's job, never the caller's.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence, TypedDict, runtime_checkable

from memforge.models import (
    DocumentRecord,
    Entity,
    EntityAlias,
    Memory,
    MemorySource,
    Project,
    SourceLifecycleResetResult,
)
from memforge.memory.evidence import (
    ActiveSupportEvidence,
    CandidateMemory,
    EvidenceReference,
    EvidenceUnit,
    MemorySupportAssertion,
    RelationOutcomeBundle,
)
from memforge.memory.lifecycle_plan import (
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
from memforge.source_projection import (
    ProjectionCoverage,
    ProjectionScopeTransition,
    SourceObservationRevision,
    SourceProjection,
    SourceUnit,
    SourceUnitInventoryFilter,
    SourceUnitInventoryPage,
    SourceUnitRevision,
)
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
from memforge.source_activity import SourceActivityLease
from memforge.storage.adapters.context import AccessScope


@dataclass(frozen=True)
class KeywordSourceRef:
    """Authorized source support row that contributed keyword evidence."""

    source_id: str | None
    doc_id: str
    source_type: str


@dataclass(frozen=True)
class KeywordCandidate:
    """Structured keyword candidate with channel and matched metadata evidence."""

    memory_id: str
    score: float
    channel: str
    matched_fields: tuple[str, ...] = ()
    source_refs: tuple[KeywordSourceRef, ...] = ()
    matched_text: tuple[str, ...] = ()


DEFAULT_ENTITY_LINK_LIMIT = 5
"""Default maximum linked entities per query; keeps graph fan-out bounded."""


@dataclass(frozen=True)
class EntityLinkCandidate:
    """Visible entity candidate with channel evidence.

    `activates_graph` is true only for linker channels trusted to seed graph
    retrieval, such as explicit, exact-alias, and lexical-alias matches.
    Diagnostic channels such as compact formatting recall can return visible
    candidates while leaving graph retrieval disabled.
    """

    entity_id: int
    canonical_name: str
    matched_alias: str
    channel: str
    contributing_channels: tuple[str, ...]
    score: float
    matched_text: str
    activates_graph: bool
    visible_memory_count: int = 0
    visible_source_count: int = 0
    specificity: float = 0.0


@dataclass(frozen=True)
class EntityLinkResult:
    """Query-time entity-linking result with unmatched explicit hints."""

    candidates: tuple[EntityLinkCandidate, ...] = ()
    unmatched_explicit_entities: tuple[str, ...] = ()


class RankingMetadata(TypedDict, total=False):
    """The per-memory inputs the ranker needs alongside RRF scores.

    `updated_at` drives the recency curve; `project_key` drives the
    cross-project affinity penalty. Both come back in one relational read
    so the ranker never makes a second roundtrip.
    """

    updated_at: datetime | None
    project_key: str | None
    repo_identifier: str | None


@runtime_checkable
class RelationalStore(Protocol):
    """Source-of-truth rows: memories and their provenance, plus the scoped
    relational channels that read those rows.

    Bound to one datastore at construction. Memory-row writes and the
    co-transactional FTS write stay inside the existing Database methods:
    this protocol delegates to those methods rather than relocating their
    SQL, preserving the single-commit atomicity that keeps SQLite and FTS5
    in sync. The read channels (graph, the post-fusion re-check, the ranking
    fetch, and the source/date re-check) own the SQL that callers run
    inline today, so a caller never reaches a database connection directly.
    """

    async def insert_memory(self, memory: Memory) -> str: ...
    async def get_memory(self, memory_id: str) -> Memory | None: ...
    async def list_active_memories(self, memory_ids: Sequence[str]) -> list[Memory]: ...
    async def list_active_candidate_memories(
        self,
        memory_ids: Sequence[str],
    ) -> list[CandidateMemory]: ...
    async def find_active_exact_claim_candidate(
        self,
        content_hash: str,
        *,
        visibility: str,
        owner_user_id: str | None,
        repo_identifier: str | None,
        excluded_memory_ids: Sequence[str] = (),
    ) -> Memory | None: ...
    async def list_active_ordinary_claim_memories(
        self,
        memory_ids: Sequence[str],
    ) -> list[Memory]: ...
    async def find_active_ordinary_claim_memories_by_entities(
        self,
        entity_ids: Sequence[int],
        *,
        visibility: str,
        owner_user_id: str | None,
        repo_identifier: str | None,
        project_key: str | None,
        excluded_memory_ids: Sequence[str] = (),
        excluded_doc_id: str | None = None,
        limit: int,
    ) -> list[Memory]: ...
    async def find_rebaseline_reactivation_candidate(
        self,
        content_hash: str,
        *,
        visibility: str,
        owner_user_id: str | None,
        repo_identifier: str | None,
    ) -> Memory | None: ...
    async def get_memory_sources(self, memory_id: str) -> list[MemorySource]: ...
    async def upsert_document(
        self,
        doc: DocumentRecord,
        *,
        require_configured_source: bool = False,
        source_activity: SourceActivityLease | None = None,
    ) -> None: ...
    async def get_document(self, doc_id: str) -> DocumentRecord | None: ...
    async def delete_projected_document(self, doc_id: str) -> None: ...
    async def rebaseline_source_lifecycle(
        self,
        source_id: str,
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> SourceLifecycleResetResult: ...
    async def rebind_projected_document_support(
        self,
        old_doc_id: str,
        new_doc_id: str,
    ) -> None: ...
    async def record_source_projection(
        self,
        projection: SourceProjection,
        *,
        expected_source_activity_epoch: int | None = None,
    ) -> None: ...
    async def get_source_projection(self, run_id: str) -> SourceProjection | None: ...
    async def get_current_source_unit_revision(
        self,
        source_unit_id: str,
    ) -> SourceUnitRevision | None: ...
    async def get_current_source_observation_revisions(
        self,
        source_unit_id: str,
    ) -> Mapping[str, SourceObservationRevision]: ...
    async def find_source_unit_by_document_id(
        self,
        source_id: str,
        document_id: str,
        *,
        current_only: bool = False,
    ) -> SourceUnit | None: ...
    async def list_source_unit_document_ids(
        self,
        source_unit_id: str,
    ) -> tuple[str, ...]: ...
    async def list_current_source_unit_observation_ids(
        self,
        source_id: str,
    ) -> dict[str, tuple[str, ...]]: ...
    async def list_current_source_units(
        self,
        source_id: str,
    ) -> tuple[SourceUnit, ...]: ...
    async def list_current_source_units_page(
        self,
        source_id: str,
        *,
        filters: SourceUnitInventoryFilter,
        cursor: str | None = None,
        limit: int = 200,
    ) -> SourceUnitInventoryPage: ...
    async def create_projection_scope_transition(
        self,
        transition: ProjectionScopeTransition,
    ) -> ProjectionScopeTransition: ...
    async def get_open_projection_scope_transition(
        self,
        source_id: str,
    ) -> ProjectionScopeTransition | None: ...
    async def list_projection_scope_transitions(
        self,
        source_id: str,
        *,
        limit: int = 20,
    ) -> list[ProjectionScopeTransition]: ...
    async def start_projection_scope_transition(
        self,
        transition_id: str,
        *,
        run_id: str,
    ) -> ProjectionScopeTransition: ...
    async def complete_projection_scope_transition(
        self,
        transition_id: str,
        *,
        run_id: str,
        coverage: ProjectionCoverage,
    ) -> ProjectionScopeTransition: ...
    async def fail_projection_scope_transition(
        self,
        transition_id: str,
        *,
        run_id: str,
        coverage: ProjectionCoverage,
        error: str,
    ) -> ProjectionScopeTransition: ...
    async def list_legacy_memory_provenance(
        self,
        source_id: str,
    ) -> list[LegacyMemoryProvenance]: ...
    async def count_active_source_memories(self, source_id: str) -> int: ...
    async def count_active_source_memories_without_support(self, source_id: str) -> int: ...
    async def get_lifecycle_gate(self, source_id: str) -> LifecycleGate: ...
    async def enable_lifecycle_gate(
        self,
        source_id: str,
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> LifecycleGate: ...
    async def gate_destructive_lifecycle(
        self,
        source_id: str,
        *,
        reason: str,
        source_activity: SourceActivityLease | None = None,
    ) -> LifecycleGate: ...
    async def upsert_lifecycle_cutover_finding(
        self,
        finding: LifecycleCutoverFinding,
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> None: ...
    async def get_lifecycle_cutover_finding(
        self,
        finding_id: str,
    ) -> LifecycleCutoverFinding | None: ...
    async def list_lifecycle_cutover_findings(
        self,
        source_id: str,
        *,
        status: CutoverFindingStatus | None = None,
    ) -> list[LifecycleCutoverFinding]: ...
    async def create_lifecycle_backfill_job(
        self,
        job: LifecycleBackfillJob,
    ) -> LifecycleBackfillJob: ...
    async def create_source_rebaseline_job(
        self,
        job: LifecycleBackfillJob,
    ) -> LifecycleBackfillJob: ...
    async def start_lifecycle_backfill_job(self, job_id: str) -> LifecycleBackfillJob: ...
    async def complete_lifecycle_backfill_job(
        self,
        job_id: str,
        *,
        scanned_memories: int,
        mapped_memories: int,
        finding_count: int,
    ) -> LifecycleBackfillJob: ...
    async def fail_lifecycle_backfill_job(
        self,
        job_id: str,
        *,
        error: str,
    ) -> LifecycleBackfillJob: ...
    async def recover_stale_lifecycle_backfill_job(
        self,
        job_id: str,
        *,
        error: str,
    ) -> LifecycleBackfillJob: ...
    async def list_stale_lifecycle_backfill_job_ids(
        self,
        *,
        limit: int = 100,
    ) -> tuple[str, ...]: ...
    async def get_lifecycle_backfill_job(self, job_id: str) -> LifecycleBackfillJob | None: ...
    async def get_active_lifecycle_backfill_job(
        self,
        source_id: str,
    ) -> LifecycleBackfillJob | None: ...
    async def list_lifecycle_backfill_jobs(
        self,
        source_id: str,
        *,
        limit: int = 20,
    ) -> list[LifecycleBackfillJob]: ...
    async def resolve_lifecycle_cutover_finding(
        self,
        finding_id: str,
        *,
        observation_id: str,
        source_unit_id: str,
        source_activity: SourceActivityLease | None = None,
    ) -> LifecycleCutoverFinding: ...
    async def retire_unprovable_lifecycle_cutover_finding(
        self,
        finding_id: str,
        *,
        source_id: str,
        reconstruction_attempt_id: str,
        operator_id: str,
        unavailable_documents: Mapping[str, str],
    ) -> LifecycleCutoverFinding: ...
    async def record_evidence_references(
        self,
        evidence_unit_id: str,
        references: Sequence[EvidenceReference],
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> tuple[EvidenceReference, ...]: ...
    async def get_evidence_unit(self, evidence_unit_id: str) -> EvidenceUnit | None: ...
    async def upsert_memory_support_assertion(
        self,
        assertion: MemorySupportAssertion,
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> None: ...
    async def get_memory_support_set_hash(self, memory_id: str) -> str: ...
    async def get_active_memory_support_reference_ids(self, memory_id: str) -> tuple[str, ...]: ...
    async def get_active_memory_support_evidence(
        self,
        memory_id: str,
        *,
        source_id: str | None = None,
    ) -> tuple[ActiveSupportEvidence, ...]: ...
    async def get_source_unit_support_reference_ids(
        self,
        source_unit_id: str,
    ) -> Mapping[str, tuple[str, ...]]: ...
    async def apply_source_projection_lifecycle(
        self,
        projection: SourceProjection,
        plan: LifecyclePlan,
        *,
        expected_source_activity_epoch: int | None = None,
    ) -> None: ...
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
    ) -> None: ...
    async def apply_lifecycle_plan(self, plan: LifecyclePlan) -> None: ...
    async def get_lifecycle_plan_payload(
        self,
        lifecycle_plan_id: str,
    ) -> Mapping[str, object] | None: ...
    async def get_lifecycle_review(self, review_id: str) -> LifecycleReview | None: ...
    async def list_lifecycle_reviews(
        self,
        source_id: str,
        *,
        status: LifecycleReviewStatus | None = None,
    ) -> list[LifecycleReview]: ...
    async def resolve_lifecycle_review(
        self,
        review_id: str,
        status: LifecycleReviewStatus,
    ) -> LifecycleReview: ...
    async def list_lifecycle_vector_tasks(
        self,
        *,
        source_id: str | None = None,
        lifecycle_plan_id: str | None = None,
        limit: int = 100,
    ) -> list[LifecycleVectorTask]: ...
    async def complete_lifecycle_vector_task(self, task_id: str) -> None: ...
    async def fail_lifecycle_vector_task(self, task_id: str, error: str) -> None: ...
    async def get_aliases_for_entity(self, entity_id: int) -> list[EntityAlias]: ...
    async def get_all_entities(self) -> list[Entity]: ...
    async def get_all_aliases(self) -> list[tuple[str, int]]: ...
    async def filter_visible_ids(self, ids: Sequence[str], scope: AccessScope) -> set[str]: ...
    async def filter_ids_by_source_and_time(
        self,
        ids: Sequence[str],
        source_filter: MemorySourceFilter | None = None,
        time_range: MemoryTimeRange | None = None,
    ) -> set[str]: ...
    async def list_ids_by_source_and_time(
        self,
        source_filter: MemorySourceFilter | None,
        time_range: MemoryTimeRange | None,
        scope: AccessScope,
        *,
        limit: int,
        offset: int,
    ) -> tuple[list[str], int]: ...
    async def fetch_ranking_metadata(self, ids: Sequence[str]) -> Mapping[str, RankingMetadata]: ...
    async def graph_search(
        self,
        entity_ids: Sequence[int],
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
        *,
        source_filter: MemorySourceFilter | None = None,
        time_range: MemoryTimeRange | None = None,
    ) -> list[tuple[str, float]]: ...
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
    ) -> EntityLinkResult: ...
    async def add_memory_source(
        self,
        memory_id: str,
        doc_id: str,
        source_type: str,
        excerpt: str | None,
        *,
        support_kind: str = "extracted",
        source_updated_at: datetime | None,
    ) -> None: ...
    async def promote_to_workspace(
        self,
        memory_id: str,
        *,
        actor_user_id: str,
        reason: str,
    ) -> None:
        """Flip a private memory to workspace visibility.

        The full promotion flow (re-stamping vector metadata in place,
        re-running dedup against the team set) is designed but not yet
        implemented. Implementations must raise NotImplementedError after
        auditing the attempt and after verifying the actor owns the row.
        A non-owner caller must be rejected before any audit emission.
        """
        ...

    async def create_project(self, *, key: str, name: str, is_shared: bool = False) -> Project: ...
    async def get_project(self, project_id: str) -> Project | None: ...
    async def list_projects(self) -> list[Project]: ...
    async def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        is_shared: bool | None = None,
    ) -> Project | None: ...
    async def list_project_memory_ids(self, project_id: str) -> list[str]:
        """Return the memory ids attached to a project.

        Pairs with `commit_project_deletion`. The handler reads the
        affected ids first, has the owning vector service rewrite their
        embedding metadata to UNSORTED, then commits the relational
        rebucket. Reserved keys (SHARED, UNSORTED) raise `ValueError`;
        an unknown id raises `LookupError`.
        """
        ...

    async def commit_project_deletion(self, project_id: str, affected_ids: Sequence[str]) -> None:
        """Rebucket the named memories to UNSORTED and drop the project
        row, in one transaction.

        `affected_ids` is the same id list the caller already moved on
        the vector side, so the relational rebucket touches exactly the
        rows the vector channel touched. Reserved keys (SHARED,
        UNSORTED) raise `ValueError`.
        """
        ...


@runtime_checkable
class KeywordSearch(Protocol):
    """BM25/FTS5 channel.

    The SQLite implementation is a thin facade: memory-row writes and their
    FTS writes stay inside the existing co-transactional Database methods, so
    this protocol owns only the read-path FTS query and the one standalone
    FTS delete.
    """

    async def remove(self, memory_id: str) -> None: ...
    async def search(
        self,
        fts_query: str,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]: ...

    async def search_metadata(
        self,
        fts_query: str,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
        *,
        source_filter: MemorySourceFilter | None = None,
        time_range: MemoryTimeRange | None = None,
        include_subchannel_hits: bool = False,
    ) -> list[KeywordCandidate]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Embedding channel. It owns every distance/score conversion so no
    caller ever assumes cosine: similarity() maps native distance to [0, 1],
    and within_dedup_threshold() decides whether a returned score is close
    enough to be a duplicate against a configured distance threshold.

    distance_metric is a declared label used only by a calibration check to
    assert thresholds match the metric. There is no metric-enum machinery:
    the single in-scope implementation is cosine.
    """

    distance_metric: str

    def similarity(self, distance: float) -> float: ...
    def within_dedup_threshold(self, distance_threshold: float, score: float) -> bool: ...
    async def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None: ...
    async def delete(self, ids: Sequence[str]) -> None: ...
    async def query(
        self,
        embedding: Sequence[float],
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]: ...
    async def get_record(self, memory_id: str) -> dict[str, Any] | None: ...
