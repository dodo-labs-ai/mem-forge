"""Convert a workspace's Cross-Source Conflict Reviews into cross-document relations.

Cross-document discovery records relations and creates no Review, so the Reviews
it created before are converted once per workspace (ADR 0037). A decided Review
converts by its relation label: ``contradicts`` when confirmed and ``none`` when
dismissed, unless the report and apply requests relabel it by Review id.

| Review | Becomes |
| --- | --- |
| labeled ``contradicts``, ``updates`` or ``equivalent``, both Memories unchanged | relation of that label decided by review |
| labeled ``none``, both Memories unchanged | Relation Dismissals of ``contradicts`` and ``updates`` for both contents |
| labeled ``none``, a Memory changed | nothing; a dismissal of old content hides nothing |
| labeled otherwise with a changed Memory, pending or stale | re-run of the challenger's discovery |

A Review converts by its Memories' content alone and never reads their Sources:
a relation and a dismissal are read under the reader's access to both Memories,
so a private, changing or deleted Source never holds back the conversion.

A Review labeled ``none`` found that both statements hold, so it dismisses both
labels that say they cannot. A person who has already undone such a dismissal
keeps that decision: the conversion never writes a dismissal again for a pair,
label and contents that one was recorded for.

Exhausted discovery work is re-run as well, and the relation rows that earlier
discovery wrote into the Evidence Unit relation projection are removed, because
only Lifecycle writes belong there.

The conversion has three separately approved steps. The report is read-only and
names its complete plan, labels included, by ``report_id``. Applying a report
computes the plan again from the same relabels, refuses a different one, and writes it in one transaction with every row
guarded by the content it was planned for; applying the same report again
returns the recorded receipt. Deleting the converted Review rows requires a
receipt whose written counts equal its planned counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Protocol

from memforge.memory.cross_document_relation import CrossDocumentRelationLabel, pair_key
from memforge.memory.cross_source_conflict_reviews import (
    CrossSourceConflictReviewStore,
    decided_review_labels,
    list_cross_source_conflict_reviews,
    review_memories_unchanged,
)
from memforge.memory.relation_discovery_contract import (
    RelationDiscoveryWork,
    RelationDiscoveryWorkSelection,
    RelationDiscoveryWorkState,
    relation_discovery_work_rerunnable,
)
from memforge.models import Memory, MemoryReview

CONVERSION_APPLIED_EVENT = "cross_source_review_conversion_applied"
CONVERTED_REVIEWS_DELETED_EVENT = "cross_source_reviews_deleted"
CONVERSION_RERUN_REASON = "cross_source_review_conversion"
# The labels a dismissed Review rejects: each says the two statements cannot both hold.
DISMISSED_REVIEW_LABELS = (
    CrossDocumentRelationLabel.CONTRADICTS,
    CrossDocumentRelationLabel.UPDATES,
)
_PAGE_SIZE = 100


class ConversionCount(str, Enum):
    """What an applied conversion writes, planned and written alike."""

    RELATIONS = "relations"
    DISMISSALS = "dismissals"
    RERUNS = "reruns"
    LEGACY_DISCOVERY_RELATIONS = "legacy_discovery_relations"


class CrossSourceReviewConversionConflict(RuntimeError):
    """The conversion cannot proceed from the state it was asked to act on."""


@dataclass(frozen=True, slots=True)
class ConvertedReviewDecision:
    """A person's decision on a Review, bound to both Memories' current content."""

    review_id: str
    label: CrossDocumentRelationLabel
    memory_low_id: str
    memory_high_id: str
    low_content_hash: str
    high_content_hash: str
    reason: str
    reviewer: str | None
    resolved_at: str | None
    note: str | None


@dataclass(frozen=True, slots=True)
class CrossSourceReviewConversionPlan:
    report_id: str
    review_ids: tuple[str, ...]
    # Decisions labeled with a relation, written as that relation.
    relations: tuple[ConvertedReviewDecision, ...]
    # Decisions labeled none, written as Relation Dismissals.
    dismissals: tuple[ConvertedReviewDecision, ...]
    discarded_review_ids: tuple[str, ...]
    rerun_review_ids: tuple[str, ...]
    # The challenger is gone, or its current discovery work has not finished and
    # will judge its pairs on its own schedule.
    nothing_to_rerun_review_ids: tuple[str, ...]
    exhausted_work_ids: tuple[str, ...]
    # Challenger work of the re-run Reviews, then exhausted work; each once.
    rerun_work_ids: tuple[str, ...]
    legacy_discovery_relation_count: int
    max_attempts: int

    @property
    def planned_counts(self) -> dict[str, int]:
        return {
            ConversionCount.RELATIONS.value: len(self.relations),
            ConversionCount.DISMISSALS.value: len(self.dismissals) * len(DISMISSED_REVIEW_LABELS),
            ConversionCount.RERUNS.value: len(self.rerun_work_ids),
            ConversionCount.LEGACY_DISCOVERY_RELATIONS.value: self.legacy_discovery_relation_count,
        }


@dataclass(frozen=True, slots=True)
class CrossSourceReviewConversionReceipt:
    report_id: str
    applied_by: str
    applied_at: str
    planned: Mapping[str, int]
    written: Mapping[str, int]
    review_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return dict(self.planned) == dict(self.written)

    def to_payload(self) -> dict[str, object]:
        return {
            "report_id": self.report_id,
            "applied_by": self.applied_by,
            "applied_at": self.applied_at,
            "planned": dict(self.planned),
            "written": dict(self.written),
            "review_ids": list(self.review_ids),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CrossSourceReviewConversionReceipt:
        return cls(
            report_id=str(payload["report_id"]),
            applied_by=str(payload["applied_by"]),
            applied_at=str(payload["applied_at"]),
            planned={str(key): int(value) for key, value in dict(payload["planned"]).items()},  # type: ignore[arg-type]
            written={str(key): int(value) for key, value in dict(payload["written"]).items()},  # type: ignore[arg-type]
            review_ids=tuple(str(item) for item in payload["review_ids"]),  # type: ignore[union-attr]
        )


class CrossSourceReviewConversionStore(CrossSourceConflictReviewStore, Protocol):
    async def list_memories_by_ids(self, memory_ids: Sequence[str]) -> list[Memory]: ...

    async def get_latest_relation_discovery_work(
        self,
        memory_id: str,
        expected_content_hash: str,
    ) -> RelationDiscoveryWork | None: ...

    async def list_relation_discovery_work(
        self,
        selection: RelationDiscoveryWorkSelection,
        *,
        limit: int,
        offset: int = 0,
    ) -> tuple[RelationDiscoveryWork, ...]: ...

    async def count_legacy_discovery_evidence_relations(self) -> int: ...

    async def get_cross_source_review_conversion(
        self,
        report_id: str,
    ) -> CrossSourceReviewConversionReceipt | None: ...

    async def apply_cross_source_review_conversion(
        self,
        plan: CrossSourceReviewConversionPlan,
        *,
        actor: str,
    ) -> CrossSourceReviewConversionReceipt: ...

    async def delete_cross_source_conflict_reviews(
        self,
        receipt: CrossSourceReviewConversionReceipt,
        *,
        actor: str,
    ) -> int: ...


async def build_conversion_plan(
    store: CrossSourceReviewConversionStore,
    *,
    label_overrides: Mapping[str, CrossDocumentRelationLabel],
    max_attempts: int,
) -> CrossSourceReviewConversionPlan:
    """Classify every Cross-Source Conflict Review and the exhausted discovery work.

    ``label_overrides`` relabels decided Reviews by id; naming any other Review
    raises ``ReviewLabelOverrideError``.
    """

    reviews = sorted(await list_cross_source_conflict_reviews(store), key=lambda review: review.id)
    labels = decided_review_labels(reviews, label_overrides)
    memories = {
        memory.id: memory
        for memory in await store.list_memories_by_ids(
            [memory_id for review in reviews for memory_id in (review.challenger_memory_id, review.incumbent_memory_id)]
        )
    }
    relations: list[ConvertedReviewDecision] = []
    dismissals: list[ConvertedReviewDecision] = []
    discarded: list[str] = []
    rerun: list[str] = []
    nothing_to_rerun: list[str] = []
    rerun_work_ids: list[str] = []
    for review in reviews:
        challenger = memories.get(review.challenger_memory_id)
        incumbent = memories.get(review.incumbent_memory_id)
        label = labels.get(review.id)
        if (
            label is not None
            and challenger is not None
            and incumbent is not None
            and review_memories_unchanged(review, challenger=challenger, incumbent=incumbent)
        ):
            decision = _converted_decision(review, label, challenger, incumbent)
            (dismissals if label is CrossDocumentRelationLabel.NONE else relations).append(decision)
            continue
        if label is CrossDocumentRelationLabel.NONE:
            discarded.append(review.id)
            continue
        work = (
            await store.get_latest_relation_discovery_work(challenger.id, challenger.content_hash)
            if challenger is not None
            else None
        )
        if work is not None and relation_discovery_work_rerunnable(work, max_attempts=max_attempts):
            rerun.append(review.id)
            rerun_work_ids.append(work.request.id)
        else:
            nothing_to_rerun.append(review.id)
    exhausted_work_ids = tuple(sorted(work.request.id for work in await _list_exhausted_work(store, max_attempts)))
    fields = {
        "review_ids": tuple(review.id for review in reviews),
        "relations": tuple(relations),
        "dismissals": tuple(dismissals),
        "discarded_review_ids": tuple(discarded),
        "rerun_review_ids": tuple(rerun),
        "nothing_to_rerun_review_ids": tuple(nothing_to_rerun),
        "exhausted_work_ids": exhausted_work_ids,
        "rerun_work_ids": tuple(dict.fromkeys((*rerun_work_ids, *exhausted_work_ids))),
        "legacy_discovery_relation_count": await store.count_legacy_discovery_evidence_relations(),
        "max_attempts": max_attempts,
    }
    return CrossSourceReviewConversionPlan(report_id=_report_id(fields), **fields)


async def apply_conversion(
    store: CrossSourceReviewConversionStore,
    *,
    report_id: str,
    label_overrides: Mapping[str, CrossDocumentRelationLabel],
    actor: str,
    max_attempts: int,
) -> CrossSourceReviewConversionReceipt:
    """Apply the reported plan once; applying the same report again returns its receipt.

    ``label_overrides`` must be the relabels the report was made with, or the
    plan differs from the report and is refused.
    """

    applied = await store.get_cross_source_review_conversion(report_id)
    if applied is not None:
        return applied
    plan = await build_conversion_plan(store, label_overrides=label_overrides, max_attempts=max_attempts)
    if plan.report_id != report_id:
        raise CrossSourceReviewConversionConflict("conversion report changed; report again before applying")
    return await store.apply_cross_source_review_conversion(plan, actor=actor)


async def delete_converted_reviews(
    store: CrossSourceReviewConversionStore,
    *,
    report_id: str,
    actor: str,
) -> int:
    """Delete the Review rows of an applied conversion whose writes all landed."""

    receipt = await store.get_cross_source_review_conversion(report_id)
    if receipt is None:
        raise CrossSourceReviewConversionConflict("conversion report has not been applied")
    if not receipt.complete:
        raise CrossSourceReviewConversionConflict(
            "applied counts differ from the report; report and apply again before deleting Reviews"
        )
    return await store.delete_cross_source_conflict_reviews(receipt, actor=actor)


def _converted_decision(
    review: MemoryReview,
    label: CrossDocumentRelationLabel,
    challenger: Memory,
    incumbent: Memory,
) -> ConvertedReviewDecision:
    by_id = {challenger.id: challenger, incumbent.id: incumbent}
    low_id, high_id = pair_key(challenger.id, incumbent.id)
    return ConvertedReviewDecision(
        review_id=review.id,
        label=label,
        memory_low_id=low_id,
        memory_high_id=high_id,
        low_content_hash=by_id[low_id].content_hash,
        high_content_hash=by_id[high_id].content_hash,
        reason=review.reason or "",
        reviewer=review.reviewer,
        resolved_at=review.resolved_at.isoformat() if review.resolved_at else None,
        note=review.review_note,
    )


def _report_id(fields: Mapping[str, object]) -> str:
    canonical = json.dumps(
        {
            key: [asdict(item) if isinstance(item, ConvertedReviewDecision) else item for item in value]
            if isinstance(value, tuple)
            else value
            for key, value in fields.items()
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "csrc-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def _list_exhausted_work(
    store: CrossSourceReviewConversionStore,
    max_attempts: int,
) -> list[RelationDiscoveryWork]:
    selection = RelationDiscoveryWorkSelection(
        state=RelationDiscoveryWorkState.EXHAUSTED,
        max_attempts=max_attempts,
    )
    work: list[RelationDiscoveryWork] = []
    while True:
        page = await store.list_relation_discovery_work(selection, limit=_PAGE_SIZE, offset=len(work))
        work.extend(page)
        if len(page) < _PAGE_SIZE:
            return work
