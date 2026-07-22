"""Storage-neutral preconditions for auditable cross-source Review writes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from memforge.memory.evidence import (
    AuthorityCase,
    LifecycleAction,
    RelationOutcomeBundle,
    RelationType,
    ReviewCase,
)
from memforge.models import MemoryReview


@dataclass(frozen=True, slots=True)
class CrossSourceReviewMemorySnapshot:
    """Memory fields required to prove a pending cross-source Review is safe."""

    memory_id: str
    status: str
    superseded_by: str | None
    updated_at: str | None
    visibility: str
    owner_user_id: str | None
    repo_identifier: str | None
    project_key: str | None


@dataclass(frozen=True, slots=True)
class CrossSourceReviewSupportSnapshot:
    """One active Support chain, normalized by a relational adapter."""

    memory_id: str
    assertion_source_id: str
    assertion_access_context_hash: str
    evidence_unit_id: str
    evidence_source_id: str
    evidence_source_lineage_id: str | None
    evidence_visibility: str
    evidence_owner_user_id: str | None
    evidence_repo_identifier: str | None
    evidence_project_key: str | None
    evidence_access_context_hash: str | None
    observation_id: str
    observation_source_id: str
    observation_revision_id: str
    current_observation_revision_id: str | None
    source_unit_id: str
    source_unit_source_id: str
    source_access_policy: str
    source_owner_user_id: str


def validate_pending_review_retry(
    requested: MemoryReview,
    existing: MemoryReview,
) -> None:
    """Fail closed unless a pending Review retry has the same immutable identity."""

    requested_identity = (
        requested.kind,
        requested.status,
        requested.incumbent_memory_id,
        requested.challenger_memory_id,
        requested.reason,
        requested.review_note,
        requested.reviewer,
        _normalized_review_time(requested.expected_incumbent_updated_at),
        _normalized_review_time(requested.expected_challenger_updated_at),
        requested.replacement_kind,
        _normalized_review_time(requested.resolved_at),
    )
    existing_identity = (
        existing.kind,
        existing.status,
        existing.incumbent_memory_id,
        existing.challenger_memory_id,
        existing.reason,
        existing.review_note,
        existing.reviewer,
        _normalized_review_time(existing.expected_incumbent_updated_at),
        _normalized_review_time(existing.expected_challenger_updated_at),
        existing.replacement_kind,
        _normalized_review_time(existing.resolved_at),
    )
    if existing.status != "pending":
        raise RuntimeError(f"memory review {existing.id} already exists with status {existing.status}")
    if requested.id != existing.id or requested_identity != existing_identity:
        raise ValueError(f"memory review {requested.id} retry identity mismatch")


def _normalized_review_time(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    normalized = str(value)
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return normalized


def validate_cross_source_review_write(
    review: MemoryReview,
    relation_outcome: RelationOutcomeBundle,
    *,
    memories: Iterable[CrossSourceReviewMemorySnapshot],
    supports: Iterable[CrossSourceReviewSupportSnapshot],
) -> None:
    """Fail closed unless one pending Review preserves two current lineages.

    Adapters own transaction isolation and row locking.  This function owns the
    provider-neutral lifecycle contract so SQLite and HANA cannot silently
    diverge on what is safe to commit.
    """

    if review.kind != "cross_source_conflict":
        return

    run = relation_outcome.relation_run
    unit = relation_outcome.evidence_unit
    relation_is_bound = any(
        relation.memory_id == review.incumbent_memory_id
        and relation.relation_type is RelationType.CONTRADICTS
        and relation.relation_run_id == run.id
        and relation.evidence_unit_id == unit.id == run.evidence_unit_id
        and relation.source_lineage_id == unit.source_lineage_id
        and relation.authority_case is AuthorityCase.CROSS_SOURCE_CONFLICT
        and not relation.is_authoritative_support
        for relation in relation_outcome.relations
    )
    if (
        review.status != "pending"
        or review.incumbent_memory_id == review.challenger_memory_id
        or run.lifecycle_action is not LifecycleAction.CREATE_REVIEW
        or run.review_case is not ReviewCase.CROSS_SOURCE_CONFLICT
        or run.evidence_unit_id != unit.id
        or not relation_is_bound
    ):
        raise ValueError("cross-source review relation contract is incomplete")

    memory_by_id = {item.memory_id: item for item in memories}
    expected_memory_ids = {
        review.incumbent_memory_id,
        review.challenger_memory_id,
    }
    if set(memory_by_id) != expected_memory_ids:
        raise ValueError("cross-source review Memory is missing")

    expected_updates = {
        review.incumbent_memory_id: review.expected_incumbent_updated_at,
        review.challenger_memory_id: review.expected_challenger_updated_at,
    }
    for memory_id, memory in memory_by_id.items():
        if memory.status != "active" or memory.superseded_by is not None:
            raise ValueError("cross-source review requires active Memories")
        expected_updated_at = expected_updates[memory_id]
        if expected_updated_at is not None and memory.updated_at != expected_updated_at:
            raise ValueError("cross-source review Memory revision changed")
        if not memory.project_key:
            raise ValueError("cross-source review requires project attribution")

    incumbent_memory = memory_by_id[review.incumbent_memory_id]
    challenger_memory = memory_by_id[review.challenger_memory_id]
    if (
        incumbent_memory.visibility != challenger_memory.visibility
        or challenger_memory.visibility != unit.visibility
        or unit.visibility not in {"workspace", "private"}
    ):
        raise ValueError("cross-source review access scope is incompatible")
    if unit.visibility == "workspace":
        if any(
            owner is not None
            for owner in (
                incumbent_memory.owner_user_id,
                challenger_memory.owner_user_id,
                unit.owner_user_id,
            )
        ):
            raise ValueError("cross-source review access scope is incompatible")
    else:
        private_owners = {
            incumbent_memory.owner_user_id,
            challenger_memory.owner_user_id,
            unit.owner_user_id,
        }
        private_repositories = {
            incumbent_memory.repo_identifier,
            challenger_memory.repo_identifier,
            unit.repo_identifier,
        }
        if None in private_owners or len(private_owners) != 1:
            raise ValueError("cross-source review access scope is incompatible")
        # Repository scope is optional for private source types.  All records
        # must still agree exactly: all-unscoped is compatible, while a mix of
        # scoped and unscoped (or different repositories) remains forbidden.
        if len(private_repositories) != 1:
            raise ValueError("cross-source review access scope is incompatible")
    if not unit.project_key:
        raise ValueError("cross-source review requires project attribution")
    if not run.access_context_hash or run.access_context_hash != unit.access_context_hash:
        raise ValueError("cross-source review access context is inconsistent")

    supports_by_memory: dict[str, list[CrossSourceReviewSupportSnapshot]] = {
        memory_id: [] for memory_id in expected_memory_ids
    }
    for support in supports:
        if support.memory_id not in supports_by_memory:
            raise ValueError("cross-source review Support references another Memory")
        supports_by_memory[support.memory_id].append(support)

    for memory_id, memory_supports in supports_by_memory.items():
        if not memory_supports:
            raise ValueError("cross-source review requires active Support")
        memory = memory_by_id[memory_id]
        for support in memory_supports:
            if (
                support.assertion_source_id != support.evidence_source_id
                or support.assertion_source_id != support.observation_source_id
                or support.assertion_source_id != support.source_unit_source_id
                or support.evidence_source_lineage_id != support.source_unit_id
            ):
                raise ValueError("cross-source review support lineage changed")
            if support.observation_revision_id != support.current_observation_revision_id:
                raise ValueError("cross-source review requires current Support")
            if (
                not support.assertion_access_context_hash
                or support.assertion_access_context_hash != support.evidence_access_context_hash
            ):
                raise ValueError("cross-source review Support access context is inconsistent")
            if (
                support.evidence_visibility != memory.visibility
                or support.source_access_policy != memory.visibility
                or support.evidence_visibility not in {"workspace", "private"}
            ):
                raise ValueError("cross-source review access scope is incompatible")
            if support.evidence_visibility == "workspace":
                if support.evidence_owner_user_id is not None:
                    raise ValueError("cross-source review access scope is incompatible")
            elif (
                support.evidence_owner_user_id != memory.owner_user_id
                or support.source_owner_user_id != memory.owner_user_id
                or support.evidence_repo_identifier != memory.repo_identifier
            ):
                raise ValueError("cross-source review access scope is incompatible")
            if not support.evidence_project_key:
                raise ValueError("cross-source review requires project attribution")

    challenger_supports = supports_by_memory[review.challenger_memory_id]
    exact_challenger_supports = [
        support
        for support in challenger_supports
        if support.evidence_unit_id == unit.id
        and support.assertion_source_id == unit.source_id
        and support.evidence_source_lineage_id == unit.source_lineage_id
        and support.assertion_access_context_hash == run.access_context_hash
    ]
    if not exact_challenger_supports:
        raise ValueError("cross-source review evidence is not active challenger Support")
    exact_challenger = exact_challenger_supports[0]
    if (
        exact_challenger.evidence_visibility != unit.visibility
        or exact_challenger.evidence_owner_user_id != unit.owner_user_id
        or exact_challenger.evidence_repo_identifier != unit.repo_identifier
        or exact_challenger.evidence_project_key != unit.project_key
    ):
        raise ValueError("cross-source review evidence is not active challenger Support")

    if not any(
        support.assertion_source_id != unit.source_id for support in supports_by_memory[review.incumbent_memory_id]
    ):
        raise ValueError("cross-source review requires distinct source lineages")
