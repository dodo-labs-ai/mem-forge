"""Pin human-labeled Memories as cross-document relation evaluation cases.

Two sets are pinned. The pair set comes from people's decisions on
Cross-Source Conflict Reviews: one case is one Memory pair. The group set comes
from labels people give candidates of discovery requests: one case is one
challenger with every candidate discovery asked about for it, in discovery's
order, and a label for some of them, so replay shows the classifier the same
candidates side by side that production showed.

A case pins every Memory as the classifier sees it, together with the
classifier contract version whose input it holds, so the set survives the
removal of the Reviews and any later change of the Memories. A case is
replayed by any classifier contract that reads the input it pinned
(``CROSS_DOCUMENT_RELATION_INPUT_VERSIONS``), and it expects the label the
program records for the human label, the same label the Review conversion
stores (``CrossDocumentRelationJudgment.recorded_label``). A case is pinned
only when every Memory it shows comes from an active workspace Source: the set
is shared workspace content, and a decision or group it cannot pin is counted
by reason, never read.
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
    CrossDocumentRelationJudgment,
    CrossDocumentRelationLabel,
    CrossDocumentRelationPair,
    RelationSubject,
    RelationSubjectStore,
    load_relation_evidence_units,
    load_relation_subjects,
)
from memforge.memory.cross_source_conflict_reviews import (
    REVIEW_DECISION_LABELS,
    CrossSourceConflictReviewStore,
    decided_review_labels,
    list_cross_source_conflict_reviews,
    review_memories_unchanged,
)
from memforge.memory.evidence import MemoryEvidenceUnitProjection
from memforge.models import Memory, MemoryReview, MemoryStatus, ReviewStatus, Visibility
from memforge.source_access import SourceAccessPolicy, SourceAccessState

RELATION_CASE_POLICY_VERSION = "cross-document-relation-cases-v3"
RELATION_CASE_GROUP_KEY = "cross_document_relation"
RELATION_GROUP_CASE_POLICY_VERSION = "cross-document-relation-group-cases-v1"
RELATION_GROUP_CASE_GROUP_KEY = "cross_document_relation_group"

# A dismissed finding is a recorded false positive; a confirmed one is a
# representative true finding.
_REVIEW_DECISION_POPULATIONS = {
    ReviewStatus.APPROVED.value: AgentEvaluationPopulation.REPRESENTATIVE_CONTROL,
    ReviewStatus.REJECTED.value: AgentEvaluationPopulation.FAILURE_REGRESSION,
}


class RelationCaseSkip(str, Enum):
    """Why a decided Review or a labelled group is not pinned."""

    # A Memory is gone, no longer active, or no longer holds the version the
    # decision was made for.
    MEMORY_CHANGED = "memory_changed"
    # A Memory or the Source it is shown from is private.
    PRIVATE_MEMORY = "private_memory"
    # The Source a Memory is shown from is gone or its access is changing.
    SOURCE_UNAVAILABLE = "source_unavailable"
    NO_SOURCE_EVIDENCE = "no_source_evidence"


class RelationGroupCaseStore(RelationSubjectStore, Protocol):
    async def list_memories_by_ids(self, memory_ids: Sequence[str]) -> list[Memory]: ...

    async def get_source(self, source_id: str) -> Mapping[str, Any] | None: ...


class RelationCaseStore(RelationGroupCaseStore, CrossSourceConflictReviewStore, Protocol):
    pass


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
    ``label_overrides`` relabels it by Review id. An ``updates`` label on a pair
    the pinned Evidence times do not order is pinned as ``contradicts``, the
    label the program records for it. Seeding again with the same
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
    pinned = await _pinned_subjects(store, challenger, (candidate,))
    if isinstance(pinned, RelationCaseSkip):
        return pinned
    unit, subjects = pinned
    expected_label = CrossDocumentRelationJudgment(
        pair=CrossDocumentRelationPair(challenger=subjects[challenger.id], candidate=subjects[candidate.id]),
        label=label,
    ).recorded_label
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
        rubric={"expected_label": expected_label.value},
        accepted_by=actor,
    )
    return _PinnedCase(
        case_id=case.case_id,
        ground_truth_revision_id=ground_truth.ground_truth_revision_id,
        label=expected_label,
        population=population,
    )


@dataclass(frozen=True, slots=True)
class RelationGroupLabels:
    """People's labels for some candidates of one challenger's discovery request.

    ``candidate_memory_ids`` lists every candidate discovery asked about for
    the challenger, in discovery's order; ``labels`` labels some of them. A
    group without labels is still replayed: the relations the classifier gives
    its candidates are counted for people to label.
    """

    challenger_memory_id: str
    candidate_memory_ids: tuple[str, ...]
    labels: Mapping[str, CrossDocumentRelationLabel]

    def __post_init__(self) -> None:
        candidates = self.candidate_memory_ids
        if not candidates:
            raise ValueError(f"group {self.challenger_memory_id} lists no candidates")
        if len(set(candidates)) != len(candidates) or self.challenger_memory_id in candidates:
            raise ValueError(
                f"group {self.challenger_memory_id} must list distinct candidates other than its challenger"
            )
        unknown = sorted(set(self.labels) - set(candidates))
        if unknown:
            raise ValueError(f"group {self.challenger_memory_id} labels Memories it does not list: {unknown}")


@dataclass(frozen=True, slots=True)
class RelationGroupSeedReport:
    cohort_id: str | None
    pinned_group_count: int
    candidate_count: int
    labelled_pair_count: int
    label_counts: Mapping[str, int]
    skipped: Mapping[str, int]
    # The reason each group that is not pinned was skipped, by challenger id.
    skipped_groups: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _PinnedGroup:
    case_id: str
    ground_truth_revision_id: str
    candidate_count: int
    labels: Mapping[str, CrossDocumentRelationLabel]


async def seed_cross_document_relation_group_cases(
    store: RelationGroupCaseStore,
    evaluation: OfflineAgentEvaluation,
    *,
    groups: Sequence[RelationGroupLabels],
    labelled_by: str,
    actor: str,
) -> RelationGroupSeedReport:
    """Pin every labelled group whose Memories can be shown and freeze one cohort.

    A group is pinned whole or not at all: replay must show the classifier
    every candidate production showed. Each label is pinned as the label the
    program records for it, so an ``updates`` label on a pair the pinned
    Evidence times do not order is pinned as ``contradicts``. Seeding again
    with the same input pins the same cases and returns the same cohort.
    """

    challenger_ids = [group.challenger_memory_id for group in groups]
    if len(set(challenger_ids)) != len(challenger_ids):
        raise ValueError("each challenger must appear in one group")
    if not labelled_by.strip():
        raise ValueError("labelled_by is required")
    pinned: list[_PinnedGroup] = []
    skipped_groups: dict[str, str] = {}
    for group in groups:
        outcome = await _pin_group_case(store, evaluation, group=group, labelled_by=labelled_by, actor=actor)
        if isinstance(outcome, RelationCaseSkip):
            skipped_groups[group.challenger_memory_id] = outcome.value
        else:
            pinned.append(outcome)
    cohort_id = None
    if pinned:
        cohort = await evaluation.freeze_cohort(
            items=[
                AgentEvaluationCohortItem(
                    case_id=case.case_id,
                    ground_truth_revision_id=case.ground_truth_revision_id,
                    population=AgentEvaluationPopulation.REPRESENTATIVE_CONTROL,
                    role=AgentEvaluationRole.RELEASE_HOLDOUT,
                    group_key=RELATION_GROUP_CASE_GROUP_KEY,
                )
                for case in pinned
            ],
            selection_policy_version=RELATION_GROUP_CASE_POLICY_VERSION,
            created_by=actor,
        )
        cohort_id = cohort.cohort_id
    label_counts = {label.value: 0 for label in CrossDocumentRelationLabel}
    for case in pinned:
        for label in case.labels.values():
            label_counts[label.value] += 1
    skipped = {reason.value: 0 for reason in RelationCaseSkip}
    for reason in skipped_groups.values():
        skipped[reason] += 1
    return RelationGroupSeedReport(
        cohort_id=cohort_id,
        pinned_group_count=len(pinned),
        candidate_count=sum(case.candidate_count for case in pinned),
        labelled_pair_count=sum(len(case.labels) for case in pinned),
        label_counts=label_counts,
        skipped=skipped,
        skipped_groups=skipped_groups,
    )


async def _pin_group_case(
    store: RelationGroupCaseStore,
    evaluation: OfflineAgentEvaluation,
    *,
    group: RelationGroupLabels,
    labelled_by: str,
    actor: str,
) -> _PinnedGroup | RelationCaseSkip:
    by_id = {
        memory.id: memory
        for memory in await store.list_memories_by_ids((group.challenger_memory_id, *group.candidate_memory_ids))
    }
    memories = [by_id.get(memory_id) for memory_id in (group.challenger_memory_id, *group.candidate_memory_ids)]
    if any(memory is None or memory.status != MemoryStatus.ACTIVE.value for memory in memories):
        return RelationCaseSkip.MEMORY_CHANGED
    challenger, *candidates = (memory for memory in memories if memory is not None)
    pinned = await _pinned_subjects(store, challenger, candidates)
    if isinstance(pinned, RelationCaseSkip):
        return pinned
    unit, subjects = pinned
    subject = subjects[challenger.id]
    expected_labels = {
        memory_id: CrossDocumentRelationJudgment(
            pair=CrossDocumentRelationPair(challenger=subject, candidate=subjects[memory_id]),
            label=label,
        ).recorded_label
        for memory_id, label in group.labels.items()
    }
    case = await evaluation.curate_case(
        case_kind=AgentEvaluationCaseKind.CROSS_DOCUMENT_RELATION_GROUP,
        source_id=unit.source_id,
        doc_id=unit.doc_id,
        source_unit_id=unit.source_unit_id,
        manifest={
            "classifier_version": CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
            "challenger": subject.to_manifest(),
            "candidates": [subjects[candidate.id].to_manifest() for candidate in candidates],
            "origin": {"labelled_by": labelled_by},
        },
        promotion_policy_version=RELATION_GROUP_CASE_POLICY_VERSION,
        created_by=actor,
    )
    ground_truth = await evaluation.accept_ground_truth(
        case_id=case.case_id,
        rubric={"expected_labels": {memory_id: label.value for memory_id, label in expected_labels.items()}},
        accepted_by=actor,
    )
    return _PinnedGroup(
        case_id=case.case_id,
        ground_truth_revision_id=ground_truth.ground_truth_revision_id,
        candidate_count=len(candidates),
        labels=expected_labels,
    )


async def _pinned_subjects(
    store: RelationGroupCaseStore,
    challenger: Memory,
    candidates: Sequence[Memory],
) -> tuple[MemoryEvidenceUnitProjection, Mapping[str, RelationSubject]] | RelationCaseSkip:
    """The challenger's Evidence Unit and every Memory's classifier input, or why they are not pinned.

    The challenger must be shown from Source Evidence, which gives the case its
    lineage. A candidate without Evidence is shown from its statement alone, as
    discovery shows it.
    """

    memories = (challenger, *candidates)
    if any(memory.visibility == Visibility.PRIVATE.value for memory in memories):
        return RelationCaseSkip.PRIVATE_MEMORY
    shown = await load_relation_evidence_units(store, memories)
    unit = shown[challenger.id]
    if unit is None or not unit.doc_id:
        return RelationCaseSkip.NO_SOURCE_EVIDENCE
    for source_id in sorted({shown_unit.source_id for shown_unit in shown.values() if shown_unit is not None}):
        skip = _source_skip(await store.get_source(source_id))
        if skip is not None:
            return skip
    return unit, await load_relation_subjects(store, memories)


def _source_skip(source: Mapping[str, Any] | None) -> RelationCaseSkip | None:
    """Why a Source keeps the Memories shown from it out of the set, if it does.

    Only an active workspace Source's content is shared with every operator of
    the workspace.
    """

    if source is None or source.get("access_state") != SourceAccessState.ACTIVE:
        return RelationCaseSkip.SOURCE_UNAVAILABLE
    if source.get("access_policy") != SourceAccessPolicy.WORKSPACE:
        return RelationCaseSkip.PRIVATE_MEMORY
    return None
