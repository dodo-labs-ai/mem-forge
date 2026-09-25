"""Pin human-labeled Memory pairs as cross-document relation evaluation cases.

The labeled set comes from people's decisions on Cross-Source Conflict Reviews.
A case pins both Memories as the classifier sees them, together with the
classifier contract version whose input it holds, so the set survives the
removal of the Reviews and any later change of either Memory, and a case is
replayed only by the contract it was pinned for. A decision is pinned only
while both Memories still hold the version it was made for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from memforge.evals.offline_evaluation import (
    AgentEvaluationCaseKind,
    AgentEvaluationCohortItem,
    AgentEvaluationPopulation,
    AgentEvaluationRole,
    OfflineAgentEvaluation,
)
from memforge.memory.cross_document_relation import (
    CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
    CrossDocumentRelationLabel,
    RelationSubjectStore,
    load_relation_subjects,
    primary_evidence_unit,
)
from memforge.memory.cross_source_conflict_reviews import (
    REVIEW_DECISION_LABELS,
    CrossSourceConflictReviewStore,
    decided_review_labels,
    list_cross_source_conflict_reviews,
    review_memories_unchanged,
)
from memforge.models import Memory, MemoryReview, ReviewStatus, Visibility

RELATION_CASE_POLICY_VERSION = "cross-document-relation-cases-v2"
RELATION_CASE_GROUP_KEY = "cross_document_relation"

# A dismissed finding is a recorded false positive; a confirmed one is a
# representative true finding.
_REVIEW_DECISION_POPULATIONS = {
    ReviewStatus.APPROVED.value: AgentEvaluationPopulation.REPRESENTATIVE_CONTROL,
    ReviewStatus.REJECTED.value: AgentEvaluationPopulation.FAILURE_REGRESSION,
}


class RelationCaseSkip(str, Enum):
    MEMORY_CHANGED = "memory_changed"
    PRIVATE_MEMORY = "private_memory"
    NO_SOURCE_EVIDENCE = "no_source_evidence"


class RelationCaseStore(RelationSubjectStore, CrossSourceConflictReviewStore, Protocol):
    async def list_memories_by_ids(self, memory_ids: Sequence[str]) -> list[Memory]: ...


@dataclass(frozen=True, slots=True)
class RelationCaseSeedReport:
    cohort_id: str | None
    pinned_case_count: int
    label_counts: Mapping[str, int]
    skipped: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _PinnedCase:
    case_id: str
    ground_truth_revision_id: str
    label: CrossDocumentRelationLabel
    population: AgentEvaluationPopulation


async def seed_cross_document_relation_cases(
    store: RelationCaseStore,
    evaluation: OfflineAgentEvaluation,
    *,
    actor: str,
    label_overrides: Mapping[str, CrossDocumentRelationLabel],
) -> RelationCaseSeedReport:
    """Pin every decided Review whose Memories are unchanged and freeze one cohort.

    A confirmed Review is labeled contradicts and a dismissed one none, unless
    ``label_overrides`` relabels it by Review id. Seeding again with the same
    input pins the same cases and returns the same cohort.
    """

    reviews = [
        review
        for status in REVIEW_DECISION_LABELS
        for review in await list_cross_source_conflict_reviews(store, status=status)
    ]
    labels = decided_review_labels(reviews, label_overrides)
    pinned: list[_PinnedCase] = []
    skipped = {reason.value: 0 for reason in RelationCaseSkip}
    for review in reviews:
        outcome = await _pin_review_case(
            store,
            evaluation,
            review=review,
            actor=actor,
            label=labels[review.id],
        )
        if isinstance(outcome, RelationCaseSkip):
            skipped[outcome.value] += 1
        else:
            pinned.append(outcome)
    cohort_id = None
    if pinned:
        cohort = await evaluation.freeze_cohort(
            items=[
                AgentEvaluationCohortItem(
                    case_id=case.case_id,
                    ground_truth_revision_id=case.ground_truth_revision_id,
                    population=case.population,
                    role=AgentEvaluationRole.RELEASE_HOLDOUT,
                    group_key=RELATION_CASE_GROUP_KEY,
                )
                for case in pinned
            ],
            selection_policy_version=RELATION_CASE_POLICY_VERSION,
            created_by=actor,
        )
        cohort_id = cohort.cohort_id
    label_counts = {label.value: 0 for label in CrossDocumentRelationLabel}
    for case in pinned:
        label_counts[case.label.value] += 1
    return RelationCaseSeedReport(
        cohort_id=cohort_id,
        pinned_case_count=len(pinned),
        label_counts=label_counts,
        skipped=skipped,
    )


async def _pin_review_case(
    store: RelationCaseStore,
    evaluation: OfflineAgentEvaluation,
    *,
    review: MemoryReview,
    actor: str,
    label: CrossDocumentRelationLabel,
) -> _PinnedCase | RelationCaseSkip:
    by_id = {
        memory.id: memory
        for memory in await store.list_memories_by_ids(
            (review.challenger_memory_id, review.incumbent_memory_id)
        )
    }
    challenger = by_id.get(review.challenger_memory_id)
    candidate = by_id.get(review.incumbent_memory_id)
    if (
        challenger is None
        or candidate is None
        or not review_memories_unchanged(review, challenger=challenger, incumbent=candidate)
    ):
        return RelationCaseSkip.MEMORY_CHANGED
    return await _pin_case(
        store,
        evaluation,
        challenger=challenger,
        candidate=candidate,
        origin={
            "review_id": review.id,
            "review_status": review.status,
            "reviewer": review.reviewer,
            "resolved_at": review.resolved_at.isoformat() if review.resolved_at else None,
        },
        label=label,
        population=_REVIEW_DECISION_POPULATIONS[review.status],
        actor=actor,
    )


async def _pin_case(
    store: RelationCaseStore,
    evaluation: OfflineAgentEvaluation,
    *,
    challenger: Memory,
    candidate: Memory,
    origin: Mapping[str, Any],
    label: CrossDocumentRelationLabel,
    population: AgentEvaluationPopulation,
    actor: str,
) -> _PinnedCase | RelationCaseSkip:
    if Visibility.PRIVATE.value in {challenger.visibility, candidate.visibility}:
        return RelationCaseSkip.PRIVATE_MEMORY
    unit = primary_evidence_unit(await store.get_memory_evidence_units(challenger.id))
    if unit is None or not unit.doc_id:
        return RelationCaseSkip.NO_SOURCE_EVIDENCE
    subjects = await load_relation_subjects(store, (challenger, candidate))
    case = await evaluation.curate_case(
        case_kind=AgentEvaluationCaseKind.CROSS_DOCUMENT_RELATION,
        source_id=unit.source_id,
        doc_id=unit.doc_id,
        source_unit_id=unit.source_unit_id,
        manifest={
            "classifier_version": CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
            "challenger": subjects[challenger.id].to_manifest(),
            "candidate": subjects[candidate.id].to_manifest(),
            "origin": dict(origin),
        },
        promotion_policy_version=RELATION_CASE_POLICY_VERSION,
        created_by=actor,
    )
    ground_truth = await evaluation.accept_ground_truth(
        case_id=case.case_id,
        rubric={"expected_label": label.value},
        accepted_by=actor,
    )
    return _PinnedCase(
        case_id=case.case_id,
        ground_truth_revision_id=ground_truth.ground_truth_revision_id,
        label=label,
        population=population,
    )
