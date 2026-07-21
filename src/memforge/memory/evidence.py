"""Storage-neutral evidence relation contracts for memory lifecycle decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Generic, Mapping, Protocol, TypeVar

from memforge.source_projection import SourceAnchor


CandidateT = TypeVar("CandidateT")


class RelationType(str, Enum):
    SUPPORTS = "supports"
    EQUIVALENT = "equivalent"
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    NO_RELATION = "no_relation"


class AuthorityCase(str, Enum):
    SAME_DOCUMENT_REVISION = "same_document_revision"
    SAME_SOURCE_LINEAGE = "same_source_lineage"
    SAME_AGENT_CLAIM = "same_agent_claim"
    SAME_PRIVATE_REPO_SCOPE = "same_private_repo_scope"
    INDEPENDENT_SUPPORT = "independent_support"
    INDEPENDENT_REFINEMENT = "independent_refinement"
    CROSS_SOURCE_CONFLICT = "cross_source_conflict"
    CROSS_SCOPE_BLOCKED = "cross_scope_blocked"


class CandidateBucket(str, Enum):
    EXACT_SOURCE_ANCHOR = "exact_source_anchor"
    SAME_DOC_LINEAGE = "same_doc_lineage"
    SAME_AGENT_CLAIM = "same_agent_claim"
    EXISTING_RELATION_GRAPH = "existing_relation_graph"
    SAME_MEMORY_SOURCE_AUTHORITY = "same_memory_source_authority"
    SHARED_ENTITIES = "shared_entities"
    SEMANTIC_VECTOR_NEIGHBORS = "semantic_vector_neighbors"
    LEXICAL_BM25 = "lexical_bm25"
    SAME_PROJECT = "same_project"
    SOURCE_TITLE_OR_TAG_OVERLAP = "source_title_or_tag_overlap"
    HYBRID_DISCOVERY = "hybrid_discovery"


class EvidenceContentProvenance(str, Enum):
    SOURCE_EXCERPT = "source_excerpt"
    LEGACY_LIMITED = "legacy_limited"
    NO_EXCERPT = "no_excerpt"


class EvidenceRole(str, Enum):
    """How a revision-pinned source reference contributes to one claim."""

    PRIMARY = "primary"
    REQUIRED = "required"
    CONTEXT = "context"


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    role: EvidenceRole
    anchor: SourceAnchor
    id: str | None = None
    evidence_unit_id: str | None = None

    @property
    def grants_support(self) -> bool:
        return self.role in {EvidenceRole.PRIMARY, EvidenceRole.REQUIRED}


@dataclass(frozen=True, slots=True)
class MemorySupportAssertion:
    id: str
    memory_id: str
    evidence_reference_id: str
    source_id: str
    access_context_hash: str
    active: bool = True
    created_at: str | None = None
    removed_at: str | None = None


@dataclass(frozen=True, slots=True)
class ActiveSupportEvidence:
    """The source excerpt and Anchor behind one active Support Assertion."""

    memory_id: str
    source_id: str
    reference_id: str
    evidence_unit_id: str
    role: EvidenceRole
    anchor: SourceAnchor
    excerpt: str | None


def validate_evidence_references(
    references: tuple[EvidenceReference, ...],
    *,
    available_revision_ids: set[str] | frozenset[str],
) -> tuple[EvidenceReference, ...]:
    """Validate extractor output before it can become lifecycle evidence."""

    if not any(item.role is EvidenceRole.PRIMARY for item in references):
        raise ValueError("evidence requires at least one PRIMARY reference")
    identities: set[tuple[str, str, str, str | None, int | None, int | None]] = set()
    for item in references:
        anchor = item.anchor
        if anchor.observation_revision_id not in available_revision_ids:
            raise ValueError(
                f"unavailable observation revision: {anchor.observation_revision_id}"
            )
        identity = (
            item.role.value,
            anchor.kind.value,
            anchor.observation_id,
            anchor.fragment_id,
            anchor.range_start,
            anchor.range_end,
        )
        if identity in identities:
            raise ValueError("duplicate evidence reference")
        identities.add(identity)
    return references


def evidence_reference_id_for(evidence_unit_id: str, reference: EvidenceReference) -> str:
    anchor = reference.anchor
    digest = sha256(
        "\x1f".join(
            [
                evidence_unit_id,
                reference.role.value,
                anchor.kind.value,
                anchor.observation_id,
                anchor.observation_revision_id,
                anchor.fragment_id or "",
                str(anchor.range_start) if anchor.range_start is not None else "",
                str(anchor.range_end) if anchor.range_end is not None else "",
            ]
        ).encode("utf-8")
    ).hexdigest()[:20]
    return f"eref-{digest}"


class LifecycleAction(str, Enum):
    NONE = "none"
    ATTACH_SUPPORT = "attach_support"
    CREATE_MEMORY = "create_memory"
    CREATE_REVISION = "create_revision"
    CREATE_REVIEW = "create_review"
    SUPERSEDE_MEMORY = "supersede_memory"
    RETIRE_MEMORY = "retire_memory"


class ReviewCase(str, Enum):
    LEGACY_LIMITED_EVIDENCE = "legacy_limited_evidence"
    MISSING_CONTENT_PROVENANCE = "missing_content_provenance"
    MULTI_DESTRUCTIVE_MATCH = "multi_destructive_match"
    EVIDENCE_UNIT_ALREADY_MATERIALIZED = "evidence_unit_already_materialized"
    MANDATORY_INCOMPLETE = "mandatory_incomplete"
    CROSS_SCOPE_BLOCKED = "cross_scope_blocked"
    NON_AUTHORITATIVE_REFINEMENT = "non_authoritative_refinement"
    MANUAL_REVIEW_GATE = "manual_review_gate"
    CROSS_SOURCE_CONFLICT = "cross_source_conflict"


MANDATORY_CANDIDATE_BUCKETS = frozenset(
    {
        CandidateBucket.EXACT_SOURCE_ANCHOR,
        CandidateBucket.SAME_DOC_LINEAGE,
        CandidateBucket.SAME_AGENT_CLAIM,
        CandidateBucket.EXISTING_RELATION_GRAPH,
        CandidateBucket.SAME_MEMORY_SOURCE_AUTHORITY,
    }
)

DESTRUCTIVE_AUTHORITY_CASES = frozenset(
    {
        AuthorityCase.SAME_DOCUMENT_REVISION,
        AuthorityCase.SAME_SOURCE_LINEAGE,
        AuthorityCase.SAME_AGENT_CLAIM,
    }
)


def is_mandatory_candidate_bucket(bucket: CandidateBucket) -> bool:
    """Whether a bucket must be checked completely before destructive lifecycle actions."""
    return bucket in MANDATORY_CANDIDATE_BUCKETS


def is_destructive_authority(authority_case: AuthorityCase) -> bool:
    """Whether an authority case can automatically mutate existing Memory lifecycle."""
    return authority_case in DESTRUCTIVE_AUTHORITY_CASES


def relation_run_id_for(
    *,
    prefix: str,
    unit: EvidenceUnit,
    action: str | LifecycleAction,
    classifier_version: str,
    candidate_memory_id: str | None = None,
    relation_type: str | RelationType | None = None,
    authority_case: str | AuthorityCase | None = None,
    bucket: str | CandidateBucket | None = None,
) -> str:
    """Stable id for one evidence relation run under a specific classifier spec.

    The Evidence Unit id makes retries of the same source observation idempotent.
    The action/classifier/candidate fields keep audits for different lifecycle
    decisions or classifier versions from overwriting one another.
    """
    action_value = action.value if isinstance(action, LifecycleAction) else action
    relation_value = relation_type.value if isinstance(relation_type, RelationType) else relation_type
    authority_value = authority_case.value if isinstance(authority_case, AuthorityCase) else authority_case
    bucket_value = bucket.value if isinstance(bucket, CandidateBucket) else bucket
    content_hash = sha256(unit.content.encode("utf-8")).hexdigest()[:16]
    digest = sha256(
        "\x1f".join(
            [
                unit.id,
                action_value,
                classifier_version,
                candidate_memory_id or "",
                relation_value or "",
                authority_value or "",
                bucket_value or "",
                unit.access_context_hash or "",
                unit.extractor_run_id or "",
                unit.source_anchor or "",
                unit.source_lineage_id or "",
                unit.doc_revision_id or "",
                content_hash,
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"relrun-{prefix}-{digest}"


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    id: str
    source_id: str
    doc_id: str | None
    doc_revision_id: str | None
    source_type: str
    source_anchor: str | None
    source_lineage_id: str | None
    project_key: str | None
    visibility: str
    owner_user_id: str | None
    repo_identifier: str | None
    content: str
    excerpt: str | None
    evidence_provenance: EvidenceContentProvenance
    client: str | None = None
    source_metadata: Mapping[str, object] = field(default_factory=dict)
    observed_at: str | None = None
    extractor_run_id: str | None = None
    access_context_hash: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateMemory:
    memory_id: str
    source_id: str | None
    doc_id: str | None
    source_lineage_id: str | None
    visibility: str
    owner_user_id: str | None
    repo_identifier: str | None
    doc_revision_id: str | None = None
    source_anchor: str | None = None
    source_metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AccessContext:
    actor_user_id: str | None
    workspace_ids: tuple[str, ...] = ()
    role_grants: tuple[str, ...] = ()
    source_subscriptions: tuple[str, ...] = ()
    repo_identifier: str | None = None
    operation_type: str | None = None


@dataclass(frozen=True, slots=True)
class RelationDecision:
    candidate_memory_id: str
    relation_type: RelationType
    authority_case: AuthorityCase
    confidence: float
    reason: str | None = None
    proposed_memory_content: str | None = None
    evidence_excerpt: str | None = None
    source_anchor: str | None = None
    entities_to_add: tuple[str, ...] = ()
    matched_bucket: CandidateBucket | None = None
    matched_bucket_complete: bool = True
    classifier_batch_key: str | None = None


@dataclass(frozen=True, slots=True)
class LifecycleDecision:
    action: LifecycleAction
    review_case: ReviewCase | None = None
    created_memory_id: str | None = None
    target_memory_id: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceRelationRecord:
    evidence_unit_id: str
    memory_id: str
    relation_type: RelationType
    authority_case: AuthorityCase
    is_authoritative_support: bool
    source_lineage_id: str | None
    confidence: float | None
    reason: str | None = None
    proposed_memory_content: str | None = None
    excerpt: str | None = None
    classifier_version: str = ""
    relation_run_id: str = ""
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class RelationRunRecord:
    id: str
    evidence_unit_id: str
    access_context_hash: str | None
    candidate_count: int
    mandatory_candidate_count: int
    checked_candidate_count: int
    incomplete_mandatory_buckets: tuple[str, ...]
    classifier_version: str | None
    lifecycle_action: LifecycleAction | None
    review_case: ReviewCase | None
    status: str
    result_memory_id: str | None = None
    audit: Mapping[str, object] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True, slots=True)
class RelationCandidateRecord:
    relation_run_id: str
    evidence_unit_id: str
    memory_id: str
    bucket: CandidateBucket
    bucket_rank: int
    candidate_rank: int
    score: float | None
    is_mandatory: bool
    bucket_complete: bool
    was_checked: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RelationOutcomeBundle:
    """Complete relation-audit write set for one lifecycle decision."""

    evidence_unit: EvidenceUnit
    relation_run: RelationRunRecord
    candidates: tuple[RelationCandidateRecord, ...] = ()
    relations: tuple[EvidenceRelationRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateBucketResult:
    bucket: CandidateBucket
    bucket_rank: int
    complete: bool
    candidates: tuple[CandidateMemory, ...]
    scores: Mapping[str, float] = field(default_factory=dict)
    candidate_reasons: Mapping[str, str] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateUniverse:
    candidates: tuple[RelationCandidateRecord, ...]
    incomplete_mandatory_buckets: tuple[str, ...]
    total_unique_candidates: int
    mandatory_candidate_count: int
    checked_candidate_count: int


def relation_candidate_retry_identity(candidate: RelationCandidateRecord) -> tuple[Any, ...]:
    return (
        candidate.relation_run_id,
        candidate.evidence_unit_id,
        candidate.memory_id,
        candidate.bucket.value,
        candidate.bucket_rank,
        candidate.candidate_rank,
        candidate.score,
        bool(candidate.is_mandatory),
        bool(candidate.bucket_complete),
        bool(candidate.was_checked),
        candidate.reason,
    )


def evidence_relation_retry_identity(relation: EvidenceRelationRecord) -> tuple[Any, ...]:
    return (
        relation.evidence_unit_id,
        relation.memory_id,
        relation.relation_type.value,
        relation.authority_case.value,
        bool(relation.is_authoritative_support),
        relation.source_lineage_id,
        relation.confidence,
        relation.reason,
        relation.proposed_memory_content,
        relation.excerpt,
        relation.classifier_version,
        relation.relation_run_id,
    )


def relation_snapshot_hash(values: tuple[tuple[Any, ...], ...] | list[tuple[Any, ...]]) -> str:
    payload = json.dumps(list(values), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def relation_bundle_snapshot_audit(
    *,
    candidates: tuple[RelationCandidateRecord, ...] | list[RelationCandidateRecord],
    relations: tuple[EvidenceRelationRecord, ...] | list[EvidenceRelationRecord],
) -> dict[str, str]:
    return {
        "candidate_snapshot_hash": relation_snapshot_hash(
            [relation_candidate_retry_identity(candidate) for candidate in candidates]
        ),
        "relation_snapshot_hash": relation_snapshot_hash(
            sorted(evidence_relation_retry_identity(relation) for relation in relations)
        ),
    }


@dataclass(frozen=True, slots=True)
class CandidatePage(Generic[CandidateT]):
    """Bounded candidate result with an explicit completeness signal."""

    candidates: tuple[CandidateT, ...]
    complete: bool
    requested_limit: int

    @property
    def returned_count(self) -> int:
        return len(self.candidates)


class MandatoryCandidateStore(Protocol):
    """Store boundary for candidate buckets that must be complete.

    These methods deliberately return ``CandidateMemory`` instead of full
    ``Memory`` rows so SQLite, HANA, and future stores expose the same
    provenance shape to the relation classifier.
    """

    async def get_candidate_memories_by_source_anchor(
        self,
        *,
        source_id: str,
        source_anchor: str,
    ) -> list[CandidateMemory]: ...

    async def get_candidate_memories_by_source_doc(
        self,
        *,
        doc_id: str,
        support_kind: str | None = None,
    ) -> list[CandidateMemory]: ...

    async def get_candidate_memories_by_agent_claim(
        self,
        *,
        claim_anchor: str,
    ) -> list[CandidateMemory]: ...

    async def get_candidate_memories_by_existing_relation_graph(
        self,
        *,
        evidence_unit_id: str,
    ) -> list[CandidateMemory]: ...


async def build_mandatory_candidate_bucket_results(
    *,
    store: MandatoryCandidateStore,
    unit: EvidenceUnit,
    access_context: AccessContext,
) -> tuple[CandidateBucketResult, ...]:
    """Load complete mandatory candidate buckets for one Evidence Unit.

    The order is part of the contract: exact source anchor, same document
    lineage, then same agent claim. Later recall-aid buckets can be appended
    by callers, but these mandatory buckets must not be capped or skipped when
    their anchor inputs are present.
    """
    buckets: list[CandidateBucketResult] = []

    def allowed(candidates: list[CandidateMemory]) -> tuple[CandidateMemory, ...]:
        return tuple(
            candidate
            for candidate in candidates
            if _source_is_visible(candidate, access_context)
            and _private_scope_is_allowed(unit, candidate, access_context)
        )

    if unit.source_anchor:
        buckets.append(
            CandidateBucketResult(
                bucket=CandidateBucket.EXACT_SOURCE_ANCHOR,
                bucket_rank=0,
                complete=True,
                candidates=allowed(
                    await store.get_candidate_memories_by_source_anchor(
                        source_id=unit.source_id,
                        source_anchor=unit.source_anchor,
                    )
                ),
                reason="exact source anchor",
            )
        )
    if unit.doc_id:
        buckets.append(
            CandidateBucketResult(
                bucket=CandidateBucket.SAME_DOC_LINEAGE,
                bucket_rank=1,
                complete=True,
                candidates=allowed(
                    await store.get_candidate_memories_by_source_doc(
                        doc_id=unit.doc_id,
                        support_kind=None,
                    )
                ),
                reason="same source document",
            )
        )
    claim_anchor = unit.source_metadata.get("claim_anchor")
    if isinstance(claim_anchor, str) and claim_anchor:
        buckets.append(
            CandidateBucketResult(
                bucket=CandidateBucket.SAME_AGENT_CLAIM,
                bucket_rank=2,
                complete=True,
                candidates=allowed(
                    await store.get_candidate_memories_by_agent_claim(
                        claim_anchor=claim_anchor,
                    )
                ),
                reason="same private agent claim",
            )
        )
    buckets.append(
        CandidateBucketResult(
            bucket=CandidateBucket.EXISTING_RELATION_GRAPH,
            bucket_rank=3,
            complete=True,
            candidates=allowed(
                await store.get_candidate_memories_by_existing_relation_graph(
                    evidence_unit_id=unit.id,
                )
            ),
            reason="existing current evidence relations",
        )
    )
    return tuple(buckets)


def build_candidate_universe(
    *,
    relation_run_id: str,
    evidence_unit_id: str,
    bucket_results: tuple[CandidateBucketResult, ...],
    recall_candidate_cap: int = 30,
) -> CandidateUniverse:
    """Build the deterministic, auditable candidate universe for one Evidence Unit.

    Mandatory buckets are never capped. Recall-aid buckets are capped after
    mandatory candidates have been preserved. Duplicate Memories keep the best
    bucket by deterministic rank so later classifier input is stable.
    """
    best_by_memory: dict[str, RelationCandidateRecord] = {}
    total_unique_ids: set[str] = set()
    recall_kept = 0

    sorted_buckets = sorted(bucket_results, key=lambda result: (result.bucket_rank, result.bucket.value))
    incomplete_mandatory_buckets = tuple(
        result.bucket.value
        for result in sorted_buckets
        if is_mandatory_candidate_bucket(result.bucket) and not result.complete
    )

    for bucket_result in sorted_buckets:
        mandatory = is_mandatory_candidate_bucket(bucket_result.bucket)
        sorted_candidates = sorted(
            enumerate(bucket_result.candidates),
            key=lambda item: (
                -float(bucket_result.scores.get(item[1].memory_id, 0.0)),
                item[0],
                item[1].memory_id,
            ),
        )
        for original_rank, candidate in sorted_candidates:
            total_unique_ids.add(candidate.memory_id)
            if not mandatory:
                if recall_kept >= recall_candidate_cap:
                    continue
                recall_kept += 1
            record = RelationCandidateRecord(
                relation_run_id=relation_run_id,
                evidence_unit_id=evidence_unit_id,
                memory_id=candidate.memory_id,
                bucket=bucket_result.bucket,
                bucket_rank=bucket_result.bucket_rank,
                candidate_rank=original_rank,
                score=bucket_result.scores.get(candidate.memory_id),
                is_mandatory=mandatory,
                bucket_complete=bucket_result.complete,
                was_checked=True,
                reason=bucket_result.candidate_reasons.get(
                    candidate.memory_id,
                    bucket_result.reason,
                ),
            )
            existing = best_by_memory.get(candidate.memory_id)
            if existing is None or (
                record.bucket_rank,
                record.candidate_rank,
                record.memory_id,
            ) < (
                existing.bucket_rank,
                existing.candidate_rank,
                existing.memory_id,
            ):
                best_by_memory[candidate.memory_id] = record

    candidates = tuple(
        sorted(
            best_by_memory.values(),
            key=lambda record: (record.bucket_rank, record.candidate_rank, record.memory_id),
        )
    )
    return CandidateUniverse(
        candidates=candidates,
        incomplete_mandatory_buckets=incomplete_mandatory_buckets,
        total_unique_candidates=len(total_unique_ids),
        mandatory_candidate_count=sum(1 for candidate in candidates if candidate.is_mandatory),
        checked_candidate_count=len(candidates),
    )


def _same_agent_claim(unit: EvidenceUnit, candidate: CandidateMemory) -> bool:
    unit_anchor = unit.source_metadata.get("claim_anchor")
    candidate_anchor = candidate.source_metadata.get("claim_anchor")
    return bool(unit_anchor and candidate_anchor and unit_anchor == candidate_anchor)


def _source_is_visible(candidate: CandidateMemory, access_context: AccessContext) -> bool:
    return candidate.source_id is None or candidate.source_id in access_context.source_subscriptions


def _private_scope_is_allowed(
    unit: EvidenceUnit,
    candidate: CandidateMemory,
    access_context: AccessContext,
) -> bool:
    if candidate.visibility != "private":
        return True
    if not access_context.actor_user_id or candidate.owner_user_id != access_context.actor_user_id:
        return False
    if candidate.repo_identifier and candidate.repo_identifier != access_context.repo_identifier:
        return False
    if unit.repo_identifier and candidate.repo_identifier and unit.repo_identifier != candidate.repo_identifier:
        return False
    return True


def classify_authority_case(
    unit: EvidenceUnit,
    candidate: CandidateMemory,
    matched_bucket: CandidateBucket,
    relation_type: RelationType,
    access_context: AccessContext,
) -> AuthorityCase:
    """Classify lifecycle authority with deterministic scope rules.

    The semantic classifier can say two items are related, but only this boundary
    decides whether the relation is allowed to affect durable Memory state.
    """
    if not _source_is_visible(candidate, access_context):
        return AuthorityCase.CROSS_SCOPE_BLOCKED
    if not _private_scope_is_allowed(unit, candidate, access_context):
        return AuthorityCase.CROSS_SCOPE_BLOCKED

    if matched_bucket is CandidateBucket.SAME_AGENT_CLAIM and _same_agent_claim(unit, candidate):
        return AuthorityCase.SAME_AGENT_CLAIM

    if (
        unit.doc_id
        and candidate.doc_id
        and unit.doc_id == candidate.doc_id
        and unit.doc_revision_id
        and candidate.doc_revision_id
        and unit.doc_revision_id == candidate.doc_revision_id
    ):
        return AuthorityCase.SAME_DOCUMENT_REVISION

    if unit.source_lineage_id and unit.source_lineage_id == candidate.source_lineage_id:
        return AuthorityCase.SAME_SOURCE_LINEAGE

    if (
        unit.visibility == "private"
        and candidate.visibility == "private"
        and unit.owner_user_id == candidate.owner_user_id == access_context.actor_user_id
        and unit.repo_identifier
        and unit.repo_identifier == candidate.repo_identifier == access_context.repo_identifier
    ):
        return AuthorityCase.SAME_PRIVATE_REPO_SCOPE

    if relation_type is RelationType.CONTRADICTS:
        return AuthorityCase.CROSS_SOURCE_CONFLICT
    if relation_type is RelationType.REFINES:
        return AuthorityCase.INDEPENDENT_REFINEMENT
    if relation_type in (RelationType.SUPPORTS, RelationType.EQUIVALENT):
        return AuthorityCase.INDEPENDENT_SUPPORT
    return AuthorityCase.CROSS_SCOPE_BLOCKED


class MemoryRelationApplyService:
    """Derive deterministic lifecycle actions from evidence relation decisions."""

    _DESTRUCTIVE_RELATIONS = {
        RelationType.EQUIVALENT,
        RelationType.REFINES,
        RelationType.CONTRADICTS,
    }

    def __init__(self, *, created_memory_ids_by_evidence_unit: dict[str, str] | None = None) -> None:
        self.created_memory_ids_by_evidence_unit = created_memory_ids_by_evidence_unit or {}

    def derive_lifecycle(
        self,
        unit: EvidenceUnit,
        decisions: list[RelationDecision],
    ) -> LifecycleDecision:
        existing_memory_id = self.created_memory_ids_by_evidence_unit.get(unit.id)
        if existing_memory_id:
            return LifecycleDecision(
                action=LifecycleAction.CREATE_REVIEW,
                review_case=ReviewCase.EVIDENCE_UNIT_ALREADY_MATERIALIZED,
                target_memory_id=existing_memory_id,
            )

        blocking_review = self._blocking_review_case(unit, decisions)
        if blocking_review is not None:
            return LifecycleDecision(action=LifecycleAction.CREATE_REVIEW, review_case=blocking_review)

        destructive = [decision for decision in decisions if self._is_destructive_candidate(decision)]
        if destructive:
            first = destructive[0]
            return LifecycleDecision(
                action=LifecycleAction.SUPERSEDE_MEMORY,
                target_memory_id=first.candidate_memory_id,
            )

        cross_source_conflicts = [
            decision
            for decision in decisions
            if decision.relation_type is RelationType.CONTRADICTS
            and decision.authority_case is AuthorityCase.CROSS_SOURCE_CONFLICT
        ]
        if cross_source_conflicts:
            return LifecycleDecision(
                action=LifecycleAction.CREATE_REVIEW,
                target_memory_id=cross_source_conflicts[0].candidate_memory_id,
            )

        if self._has_attachable_support(decisions):
            return LifecycleDecision(action=LifecycleAction.ATTACH_SUPPORT)

        created_memory_id = self._memory_id_for_unit(unit)
        self.created_memory_ids_by_evidence_unit[unit.id] = created_memory_id
        return LifecycleDecision(action=LifecycleAction.CREATE_MEMORY, created_memory_id=created_memory_id)

    def _blocking_review_case(
        self,
        unit: EvidenceUnit,
        decisions: list[RelationDecision],
    ) -> ReviewCase | None:
        if any(decision.authority_case is AuthorityCase.CROSS_SCOPE_BLOCKED for decision in decisions):
            return ReviewCase.CROSS_SCOPE_BLOCKED

        if any(self._is_destructive_candidate(decision) for decision in decisions):
            if unit.evidence_provenance is EvidenceContentProvenance.LEGACY_LIMITED:
                return ReviewCase.LEGACY_LIMITED_EVIDENCE
            if unit.evidence_provenance is not EvidenceContentProvenance.SOURCE_EXCERPT or not unit.excerpt:
                return ReviewCase.MISSING_CONTENT_PROVENANCE
            destructive_decisions = [decision for decision in decisions if self._is_destructive_candidate(decision)]
            if any(not decision.matched_bucket_complete for decision in destructive_decisions):
                return ReviewCase.MANDATORY_INCOMPLETE
            destructive_targets = {
                decision.candidate_memory_id
                for decision in destructive_decisions
            }
            if len(destructive_targets) > 1:
                return ReviewCase.MULTI_DESTRUCTIVE_MATCH

        if any(decision.proposed_memory_content for decision in decisions):
            if unit.evidence_provenance is not EvidenceContentProvenance.SOURCE_EXCERPT or not unit.excerpt:
                return ReviewCase.MISSING_CONTENT_PROVENANCE

        if any(
            decision.relation_type is RelationType.REFINES
            and decision.authority_case is AuthorityCase.INDEPENDENT_REFINEMENT
            for decision in decisions
        ):
            return ReviewCase.NON_AUTHORITATIVE_REFINEMENT

        return None

    def _is_destructive_candidate(self, decision: RelationDecision) -> bool:
        return (
            decision.relation_type in self._DESTRUCTIVE_RELATIONS
            and is_destructive_authority(decision.authority_case)
        )

    @staticmethod
    def _has_attachable_support(decisions: list[RelationDecision]) -> bool:
        return any(
            decision.relation_type in (RelationType.SUPPORTS, RelationType.EQUIVALENT, RelationType.REFINES)
            and decision.authority_case
            in {
                AuthorityCase.SAME_DOCUMENT_REVISION,
                AuthorityCase.SAME_SOURCE_LINEAGE,
                AuthorityCase.SAME_AGENT_CLAIM,
                AuthorityCase.SAME_PRIVATE_REPO_SCOPE,
                AuthorityCase.INDEPENDENT_SUPPORT,
                AuthorityCase.INDEPENDENT_REFINEMENT,
            }
            for decision in decisions
        )

    @staticmethod
    def _memory_id_for_unit(unit: EvidenceUnit) -> str:
        digest = sha256(f"{unit.id}\x1f{LifecycleAction.CREATE_MEMORY.value}".encode("utf-8")).hexdigest()[:16]
        return f"mem-{digest}"
